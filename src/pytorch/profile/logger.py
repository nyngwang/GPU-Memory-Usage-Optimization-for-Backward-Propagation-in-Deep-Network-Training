import torch

logfile = None


def create_logfile(filename):
    global logfile
    logfile = open(filename, "w")


def log(str):
    if logfile:
        logfile.write("{}\n".format(str))
        logfile.flush()
        # The `flush()` method is used to force the output of any buffered data to be written to the file. In the `log` function,
        # `logfile.flush()` is called after writing a log message to the file to ensure that the message is immediately written to the file and not held in a buffer.
        # This is important because if the program crashes or terminates unexpectedly, any buffered data may be lost. By calling `flush()`,
        # we ensure that the log message is immediately written to the file and not lost in case of a crash or unexpected termination.
    else:
        print(str)


def cuda_memory_usage():
    allocated = torch.cuda.memory_allocated(None) // 1024**2
    reserved = torch.cuda.memory_reserved(None) // 1024**2
    return allocated, reserved


