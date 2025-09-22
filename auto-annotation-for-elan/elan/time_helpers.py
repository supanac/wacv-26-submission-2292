from os import path

def filename2timestamps(fn):
    fn_basename = path.basename(fn)
    splitted_basename = fn_basename.split("-")
    start_timestamp = splitted_basename[-2]
    end_timestamp = splitted_basename[-1]
    end_timestamp = ".".join(end_timestamp.split(".")[:-1])
    return float(start_timestamp), float(end_timestamp)

def timestamp2frame(timestamp, fps):
    return round(timestamp * fps)

def filename2frames(fn, fps):
    if fn == "":
        start_frame = 0
        end_frame = None
    else:
        start_timestamp, end_timestamp = filename2timestamps(fn)
        start_frame = timestamp2frame(start_timestamp, fps)
        end_frame = timestamp2frame(end_timestamp, fps)
    return start_frame, end_frame