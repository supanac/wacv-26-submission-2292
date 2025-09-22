from typing import Optional

import numpy as np

from .kernel import Kernel
from .io_helpers import load_all_jsons, load_scenes
from .linear_interpolation import LinearInterpolation
from .datatypes import (
    Tracks, Scores, NumpyArray, Index, StrPath, SceneList,
    ActiveSpeakerDict
)


class SpeakerKeypoints:
    """This class is aimed to store coordinates of OpenPose keypoints.

    `SpeakerKeypoints` could comprise as many instance of `PoseTimeSeries` 
    as we want. Each of the `PoseTimeSeries` instances stores coordinates of
    keypoints for a given part of body (in pose, face, or hands keypoints).

    Args:
        scores: Scores from the active speaker detection pipeline.
        tracks: Tracks from the active speaker detection pipeline.
        all_jsons: List of all dictionaries loaded from OpenPose json files.
        fps: Original frame per second ratio.

    Methods:
        append_data_to_track: Adds necessary data to active speakers' tracks.
            ASD pipeline produces an output in 25 fps ratio. This method
            interpolates needed data in order to be able to use it with 
            the original fps.
        init_keypoints: For every frame finds an active speaker and store
            active speaker's coordinates in `PoseTimeSeries` instances.
    """
    def __init__(
            self, 
            scores: Scores,
            tracks: Tracks,
            path_to_jsons: list,
            fps: float,
            video_width: int,
            video_height: int
            ) -> None:
        self.scores = scores
        self.tracks = tracks
        self.fps = fps
        self.video_width = video_width
        self.video_height = video_height
        self.all_data_from_jsons = np.array(load_all_jsons(path_to_jsons))
        self.num_jsons = len(self.all_data_from_jsons)
        self.pose_keypoints_2d = PoseTimeSeries(
            series_len=self.num_jsons, 
            num_keypoints=25, 
            descr="pose_keypoints_2d_raw", 
            dim=3
            )
        self.pose_keypoints_2d_norm = PoseTimeSeries(
            series_len=self.num_jsons, 
            num_keypoints=25, 
            descr="pose_keypoints_2d_norm", 
            dim=2
            )
        self.face_keypoints_2d = PoseTimeSeries(
            series_len=self.num_jsons, 
            num_keypoints=70, 
            descr="face_keypoints_2d_raw", 
            dim=3
        )
        self.face_keypoints_2d_norm = PoseTimeSeries(
            series_len=self.num_jsons, 
            num_keypoints=70, 
            descr="face_keypoints_2d_norm", 
            dim=2
        )
        self.right_hand_keypoints_2d = PoseTimeSeries(
            series_len=self.num_jsons, 
            num_keypoints=21, 
            descr="right_hand_keypoints_2d_raw", 
            dim=3
        )
        self.right_hand_keypoints_2d_norm = PoseTimeSeries(
            series_len=self.num_jsons, 
            num_keypoints=21, 
            descr="right_hand_keypoints_2d_norm", 
            dim=2
        )
        self.left_hand_keypoints_2d = PoseTimeSeries(
            series_len=self.num_jsons, 
            num_keypoints=21, 
            descr="left_hand_keypoints_2d_raw", 
            dim=3
        )
        self.left_hand_keypoints_2d_norm = PoseTimeSeries(
            series_len=self.num_jsons, 
            num_keypoints=21, 
            descr="left_hand_keypoints_2d_norm", 
            dim=2
        )
        all_indices, all_scores = self.find_active_speaker_indices_and_scores()
        all_tracks = self.find_active_speakers_tracks(all_indices, all_scores)
        self.save_keypoints(all_tracks)
        self.interpolate_and_normalise(all_tracks)
        self.all_tracks = all_tracks

    def save_keypoints(self, all_tracks):
        for track in all_tracks.values():
            start_frame_25_fps, end_frame_25_fps = track["boundaries"]
            track_bboxes = track["bboxes"]
            for i, frame_num_25_fps in enumerate(range(start_frame_25_fps, end_frame_25_fps)):
                if self.fps != 25:
                    frame_num = self.convert_frame_num(frame_num_25_fps, 25, self.fps)
                else:
                    frame_num = frame_num_25_fps
                frame_json = self.all_data_from_jsons[frame_num]
                frame_bbox = track_bboxes[i]
                person_json_data = self.find_keypoints_by_bbox(frame_json, frame_bbox)
                if person_json_data is not None:
                    self.store_values(person_json_data, frame_num)
        return None
        
    def convert_frame_num(self, frame_num, origin_fps, target_fps):
        return round(frame_num / origin_fps * target_fps)
    
    def find_active_speaker_indices_and_scores(self):
        all_scores = list()
        all_indices = list()
        start_frame = 0
        end_frame = len(self.all_data_from_jsons)
        if self.fps != 25:
            end_frame = int(end_frame / self.fps * 25)
        for frame_num in range(start_frame, end_frame):
            # frame_num here is in 25 fps ratio.
            tracks_include_frame = [
                (i, track["track"]["frame"])
                for i, track in enumerate(self.tracks) 
                if frame_num in track["track"]["frame"][:-1]
            ]
            where_frame_per_tracks = [
                frame_num - track_frames[1][0] for track_frames in tracks_include_frame
            ]
            inds_tracks_include_frame = [
                elem[0] for elem in tracks_include_frame
            ]
            scores_for_frame = [
                self.scores[ind][track_ind] 
                for ind, track_ind 
                in zip(inds_tracks_include_frame, where_frame_per_tracks)
                ]
            if len(scores_for_frame) == 0:
                all_scores.append(None)
                all_indices.append(None)
                continue
            top = max(scores_for_frame).item()
            scores_argmax = np.argmax(scores_for_frame)
            track_index = inds_tracks_include_frame[scores_argmax]
            all_scores.append(top)
            all_indices.append(track_index)
        return all_indices, all_scores

    def find_active_speakers_tracks(self, all_indices, all_scores):
        # Here all frame numbers are in 25 fps ratio, too.
        all_tracks = dict()
        start_segment = 0
        end_segment = start_segment
        track_ind = 0
        if self.fps != 25:
            end_frame = int(self.num_jsons / self.fps * 25)
        else:
            end_frame = self.num_jsons
        while end_segment < end_frame:
            if all_indices[start_segment] is None:
                start_segment += 1
                end_segment = start_segment
            elif all_indices[end_segment] == all_indices[start_segment]:
                end_segment += 1
            else:
                segment_scores = all_scores[start_segment:end_segment]
                segment_scores_avg = sum(segment_scores) / len(segment_scores)
                if segment_scores_avg > 0 and end_segment - start_segment > 1:
                    speaker_track_ind = all_indices[start_segment]
                    speaker_track_frames = self.tracks[speaker_track_ind]["track"]["frame"]
                    speaker_track_bboxes = self.tracks[speaker_track_ind]["track"]["bbox"]
                    speaker_segment_start_frame = start_segment - speaker_track_frames[0]
                    speaker_segment_end_frame = end_segment - speaker_track_frames[0]
                    speaker_segment_bboxes = speaker_track_bboxes[speaker_segment_start_frame:speaker_segment_end_frame]
                    speaker_segment_bboxes[:, [0, 2]] = np.clip(speaker_segment_bboxes[:, [0, 2]], 0, self.video_width)
                    speaker_segment_bboxes[:, [1, 3]] = np.clip(speaker_segment_bboxes[:, [1, 3]], 0, self.video_height)
                    all_tracks[track_ind] = {
                        "boundaries": (start_segment, end_segment),
                        "bboxes": speaker_segment_bboxes,
                        "track_index": speaker_track_ind
                    }
                    track_ind += 1
                start_segment = end_segment
        return all_tracks

    def store_values(self, person_json_data, frame_num) -> None:
        pose_kpts = np.array(person_json_data["pose_keypoints_2d"]).reshape(-1,3)
        face_kpts = np.array(person_json_data["face_keypoints_2d"]).reshape(-1,3)
        right_hand_kpts = np.array(person_json_data["hand_right_keypoints_2d"]).reshape(-1,3)
        left_hand_kpts = np.array(person_json_data["hand_left_keypoints_2d"]).reshape(-1,3)
        self.pose_keypoints_2d(frame_num, pose_kpts, False)
        self.face_keypoints_2d(frame_num, face_kpts, False)
        self.right_hand_keypoints_2d(frame_num, right_hand_kpts, False)
        self.left_hand_keypoints_2d(frame_num, left_hand_kpts, False)
        return None

    def interpolate_and_normalise(self, all_tracks) -> None:
        self.pose_keypoints_2d_norm.pose_time_series = self.interpolate_and_normalise_ts("pose", all_tracks)
        self.face_keypoints_2d_norm.pose_time_series = self.interpolate_and_normalise_ts("face", all_tracks)
        self.right_hand_keypoints_2d_norm.pose_time_series = self.interpolate_and_normalise_ts("right_hand", all_tracks)
        self.left_hand_keypoints_2d_norm.pose_time_series = self.interpolate_and_normalise_ts("left_hand", all_tracks)

    def interpolate_and_normalise_ts(self, level: str, all_tracks: list) -> NumpyArray:
        if level == "pose":
            normalisation = self.pose_keypoints_2d.normalise
            data = self.pose_keypoints_2d.pose_time_series
        elif level == "face":
            normalisation = self.face_keypoints_2d.normalise
            data = self.face_keypoints_2d.pose_time_series
        elif level == "right_hand":
            normalisation = self.right_hand_keypoints_2d.normalise
            data = self.right_hand_keypoints_2d.pose_time_series
        elif level == "left_hand":
            normalisation = self.left_hand_keypoints_2d.normalise
            data = self.left_hand_keypoints_2d.pose_time_series
        interpolated_data = self.interpolate(data, all_tracks)
        return normalisation(interpolated_data, level, all_tracks, self.fps)
    
    def interpolate(self, data, all_tracks):
        interpolated_data = np.empty_like(data)
        interpolated_data = np.copy(data)
        for track in all_tracks.values():
            start_segment_25_fps, end_segment_25_fps = track["boundaries"]
            if self.fps != 25:
                start_segment = self.convert_frame_num(start_segment_25_fps, 25, self.fps)
                end_segment = self.convert_frame_num(end_segment_25_fps, 25, self.fps)
            else:
                start_segment = start_segment_25_fps
                end_segment = end_segment_25_fps
            interpolated_data[start_segment:end_segment] = np.nan_to_num(interpolated_data[start_segment:end_segment])
            LinearInterpolation.linear_interpolation(interpolated_data[start_segment:end_segment])
        return interpolated_data
    
    def find_active_speaker_dev(self, frame: int) -> ActiveSpeakerDict:
        ## Find the tracks that include the given frame.
        #+ Keep them in a list of tuples:
        #+ [(ind_0, frames_0), (ind_1, frames_1), ... (ind_N, frames_n)]
        tracks_include_frame = [
            (i, track["track"]["frame"])
            for i, track in enumerate(self.tracks) 
            if frame in track["track"]["frame"][:-1]
        ]
        # If no track found, return None
        if not tracks_include_frame:
            return None, None, None
        ## Different tracks could start at a different frame number.
        #+ To handle this, store indices of a given frame for all the trakcs.
        where_frame_per_tracks = [
            frame - track_frames[1][0] for track_frames in tracks_include_frame
        ]
        # Store the indices of tracks that contain the given frame.
        inds_tracks_include_frame = [
            elem[0] for elem in tracks_include_frame
        ]
        try:
            ## Create a list of TalkNet scores 
            #+ for all tracks that contain the given frame number.
            scores_for_frame = [
                self.scores[ind][track_ind] 
                for ind, track_ind 
                in zip(inds_tracks_include_frame, where_frame_per_tracks)
                ]
        except:
            return None, None, None
        if not scores_for_frame:
            return None, None, None
        ## Return the bounding box that correspond to the scores' argmax.
        scores_argmax = np.argmax(scores_for_frame)
        track_index = inds_tracks_include_frame[scores_argmax]
        frame_index = where_frame_per_tracks[scores_argmax]
        bbox = self.tracks[track_index]["track"]["bbox"][frame_index]
        return bbox, max(scores_for_frame), track_index
        
    def find_active_speaker(self, frame: int) -> ActiveSpeakerDict:
        """Find an active speaker for a given frame.
        
        Args:
            frame: frame number.

        Returns:
            score (optional): A score of an active speaker, if the score is
                positive, None otherwise.
            bbox (optional): A bounding box of an active speaker, if a speaker
                was found, None otherwise.
            asd_kpts (optional): Interpolated coordinates of an active speaker,
                if the speaker was found, None otherwise.
            normed_kpts (optional): Normalised coordinates of an active
                speaker, if the speaker was found, None otherwise.
        """
        tracks_include_frame = [
            (i, track["track"]["frame"])
            for i, track in enumerate(self.tracks) 
            if frame in track["track"]["frame"][:-1]
        ]
        if not tracks_include_frame:
            return None
        where_frame_per_tracks = [
            frame - track_frames[1][0] for track_frames in tracks_include_frame
        ]
        inds_tracks_include_frame = [
            elem[0] for elem in tracks_include_frame
        ]
        try:
            scores_for_frame = [
                self.tracks[ind]["scores"][track_ind] 
                for ind, track_ind in zip(inds_tracks_include_frame, where_frame_per_tracks)
            ]
        except:
            return None
        if not scores_for_frame:
            return None
        scores_argmax = np.argmax(scores_for_frame)
        track_index = inds_tracks_include_frame[scores_argmax]
        frame_index = where_frame_per_tracks[scores_argmax]
        score = self.tracks[track_index]["scores"][frame_index]
        bbox = self.tracks[track_index]["bboxes"][frame_index]
        asd_pose_kpts = self.tracks[track_index]["pose_coordinates"][frame_index]
        asd_face_kpts = self.tracks[track_index]["face_coordinates"][frame_index]
        asd_right_hand_kpts = self.tracks[track_index]["right_hand_coordinates"][frame_index]
        asd_left_hand_kpts = self.tracks[track_index]["left_hand_coordinates"][frame_index]
        if score > 0:
            return {
                    "score": score,
                    "bbox": bbox,
                    "asd_pose_kpts": asd_pose_kpts,
                    "asd_face_kpts": asd_face_kpts,
                    "asd_right_hand_kpts": asd_right_hand_kpts,
                    "asd_left_hand_kpts": asd_left_hand_kpts,
                    "track_index": track_index
                }
        else:
            return None

    def find_keypoints_by_bbox(self, keypoints: dict, bbox: NumpyArray) -> Optional[dict]:
        """Find a person in OpenPose json file given a bounding box

        If a nose of a person from `keypoints` lies inside a bounding box,
        then we found a desired person.

        Args:
            keypoints: A dictionary loaded from a json file.
            bbox: Coordinates of a bounding box.

        Returns:
            person (optional): Dictionary with coordinates, if a person was 
                found. None otherwise.
        """
        people = keypoints['people']
        if not people or bbox is None:
            return None
        for person in people:
            person_kpts = person['pose_keypoints_2d']
            head_x, head_y = person_kpts[:2]
            if (bbox[0] <= head_x <= bbox[2]) and (bbox[1] <= head_y <= bbox[3]):
                return person
            
    def get_data(self) -> dict:
        """Returns interpolated and normalised coordinates of active speakers."""
        ret_dict = {
            "pose_keypoints_2d": self["pose_keypoints_2d"],
            "pose_keypoints_2d_norm": self["pose_keypoints_2d_norm"],
            "face_keypoints_2d": self["face_keypoints_2d"],
            "face_keypoints_2d_norm": self["face_keypoints_2d_norm"],
            "hand_right_keypoints_2d": self["right_hand_keypoints_2d"],
            "hand_right_keypoints_2d_norm": self["right_hand_keypoints_2d_norm"],
            "hand_left_keypoints_2d": self["left_hand_keypoints_2d"],
            "hand_left_keypoints_2d_norm": self["left_hand_keypoints_2d_norm"]
        }
        ret_dict = {key: val.pose_time_series for key, val in ret_dict.items()}
        ret_dict.update(
            {
                # "track_indices": self["track_indices"],
                "confidence_scores": self.pose_keypoints_2d.confidences,
                "boundaries": [elem["boundaries"] for elem in self.all_tracks.values()],
                "track_index": [elem["track_index"] for elem in self.all_tracks.values()]
            }
        )
        return ret_dict
    
    def __getitem__(self, field):
        assert field in self.__dict__
        return self.__dict__[field]
    
    def __repr__(self) -> str:
        repr_str = (
            self.pose_keypoints_2d.__repr__() 
            + self.hand_left_keypoints_2d.__repr__()
            + self.hand_right_keypoints_2d.__repr__()
            + self.face_keypoints_2d.__repr__()
        )
        return repr_str
    

class PoseTimeSeries:
    """
    This class is developed to store time series of keypoints.
    They might be {interpolated,normalised} keypoints for {posture,face,hands}.

    It has methods that allows one to set and get values of time series and 
    normalise their values.
    
    Args:
        series_len: The length of the time series. 
            Equals to the number of frame in a video.
        num_keypoints: The number of keypoints for the time series.
        descr (optional): The name of the time series.
        kernel_type (optional): The kernel type for smoothing the time series.
        kernel_size (optional): The size of the kernel.

    Attributes:
        dim (int): How many values will we keep for each keypoint.
            Default: 2, namely, x and y coodrinates.
        pose_time_series (NumpyArray): Numpy array which stores keypoints 
            coordinates.
        confidence (NumpyArray): Numpy array which stores confidence scores
            for each keypoint.
    """
    def __init__(
            self, 
            series_len: int,
            num_keypoints: int, 
            descr: str="",
            kernel_type: str="epanechnikov",
            kernel_size: int=11,
            dim: int=2,
            ) -> None:
        self.descr = descr
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.num_keypoints = num_keypoints
        self.pose_time_series = np.empty((series_len, num_keypoints, dim))
        self.pose_time_series.fill(np.nan)
        self.confidences = np.empty((series_len, num_keypoints))
        self.confidences.fill(np.nan)
        self.kernel = Kernel(kernel_type, kernel_size)

    def set_val(self, i: int, frame_keypoints: NumpyArray, normed: bool) -> None:
        """Sets keypoints coordinates for a given frame.
        
        Args:
            i: Frame number.
            frame_keypoints: Keypoints coordinates.
            normed: Are `frame_coodinates` normalised (True) or 
                interpolated (False).
        """
        self.pose_time_series[i] = frame_keypoints
        if frame_keypoints.shape[-1] == 3:
            self.confidences[i] = frame_keypoints[..., 2]
        return None
    
    def normalise(self, data: NumpyArray, level: str, all_tracks: list, video_fps: float) -> NumpyArray:
        return self._normalise(data, level, all_tracks, video_fps)
    
    def _normalise(self, data: NumpyArray, level: str, all_tracks: list, video_fps: float) -> NumpyArray:
        """Normalises values of interpolated keypoints coordinates.

        Given interpolated coordinates, this method extracts frame-wise
        coordinates of the origin and a reference point, both depend on 
        `level. 
        `level` distinguish between several cases:
            `pose`: Normalise for pose keypoints.
                Origin: Nose.
                Reference point: Neck.
            `face`: Normalise for hand keypoints.
                Origin: The tip of a nose.
                Reference point: The bridge of a nose.
            `hand`: No normalisation.
        Origin coordinates are subtracted from `data` and the difference is
        divided by a normalisation unit, i.e. a distance between the origin
        and the reference point. Frame-wise normalisation is used.

        Args:
            data: Interpolated keypoints coordinates.
            level: Describes for which keypoints normalisation is applied.

        Returns:
            Numpy array of normalised values.
        """
        if level == "pose":
            origin_keypt = 0
            reference_keypt = 1
        elif level == "face":
            origin_keypt = 30
            reference_keypt = 27
        elif "hand" in level:
            return data[..., :-1]
        normed = np.empty_like(data)
        normed.fill(np.nan)
        normed[..., -1] = data[..., -1]
        for track in all_tracks.values():
            scene_sf, scene_ef = track["boundaries"]
            if video_fps != 25:
                scene_sf = round(scene_sf / 25 * video_fps)
                scene_ef = round(scene_ef / 25 * video_fps)
            if scene_ef - scene_sf < 6:
                continue
            data[scene_sf:scene_ef] = self.kernel(data[scene_sf:scene_ef])
            scene_data = data[scene_sf:scene_ef, :, :-1]
            origin = scene_data[:, origin_keypt]
            reference = scene_data[:, reference_keypt]
            origin_non_nans_inds = ~np.isnan(origin.sum(axis=1))
            reference_non_nans_inds = ~np.isnan(reference.sum(axis=1))
            assert np.all(origin_non_nans_inds == reference_non_nans_inds)
            non_nans_inds = np.arange(len(reference))[reference_non_nans_inds]
            # origin = origin[non_nans_inds].mean(axis=0)
            # reference = reference[non_nans_inds].mean(axis=0)
            scene_norm_unit = self._calc_normalisation_unit(origin, reference)
            _, num_series, _ = scene_data.shape
            for i in range(num_series):
                if scene_data[:, i].sum() == 0: # Another condition?
                    continue
                normed[scene_sf:scene_ef, i, :-1] = (scene_data[:, i] - origin) / scene_norm_unit
                # num = scene_data[:, i] - np.nan_to_num(origin)
                # denom = np.nan_to_num((np.linalg.norm(origin - reference, axis=1) + 1e-4), nan=1)
                # normed[scene_sf:scene_ef, i, :-1] = num / denom[:, None]
        normed[..., :2] = normed[..., :2] - normed[:, 0, :2][:, None, :]
        return normed

    def _calc_normalisation_unit(self, origin: NumpyArray, reference: NumpyArray) -> NumpyArray:
        """Calculates normalisation units frame-wise (euclidian distance).
        
        Args:
            origin: Origin points.
            reference: Reference points

        Returns:
            Frame-wise distances between origin and reference points.
        """
        eps = 1e-4
        return np.linalg.norm(origin - reference) + eps

    def __call__(self, i: int, keypoints: list, normed: bool) -> None:
        self.set_val(i, keypoints, normed)
        return None
    
    def __getitem__(self, idx: Index) -> NumpyArray:
        if isinstance(idx, int):
            return self.pose_time_series[idx]
        elif isinstance(idx, slice):
            return self.pose_time_series[idx.start:idx.stop:idx.step]
        
    def __setitem__(self, idx: Index, value) -> None:
        if isinstance(idx, int):
            self.pose_time_series[idx] = value
        elif isinstance(idx, slice):
            self.pose_time_series[idx.start:idx.stop:idx.step] = value

    def __repr__(self) -> str:
        ts_len, num_keypoints, _ = self.pose_time_series.shape
        repr_str = (
            f"Contains keypoints \"{self.descr}\";\n"
            f"Time Series length is {ts_len:d};\n"
            f"Number of keypoints is {num_keypoints:d}\n"
        )
        return repr_str