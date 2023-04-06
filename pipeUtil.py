from pipe import Pipe

@Pipe
def pipable(func, arg):
    func(arg)