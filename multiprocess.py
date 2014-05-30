__author__ = 'adeb'

import multiprocessing


def spawn(f):
    def fun(q_in, q_out):
        while True:
            i, x = q_in.get()
            if i is None:
                break
            q_out.put((i, f(x)))
    return fun


def parmap(f, x, nprocs=multiprocessing.cpu_count()):
    """
    Parallel map that can be used with method functions or lambda functions contrary to the built multiprocessing map
    or imap functions.
    """
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=spawn(f), args=(q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(x)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]