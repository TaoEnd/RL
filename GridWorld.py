# coding:utf-8

import numpy as np
import sys
# discrete是用来构建离散空间的，box是用来构建连续空间的
from gym.envs.toy_text import discrete
from io import StringIO

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridWorldEnv(discrete.DiscreteEnv):
    """
        Grid World environment from Sutton's Reinforcement Learning book chapter 4.
        You are an agent on an MxN grid and your goal is to reach the terminal
        state at the top left or the bottom right corner.

        For example, a 4x4 grid looks as follows:

        T  o  o  o
        o  x  o  o
        o  o  o  o
        o  o  o  T

        x is your position and T are the two terminal states.

        You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
        Actions going off the edge leave you in your current state.
        You receive a reward of -1 at each step until you reach a terminal state.
    """
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, shape=[4, 4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError("shape argument must be a list/tuple of length 2")

        self.shape = shape
        ns = np.prod(shape)   # np.prob()默认情况下计算所有元素的乘积
        na = 4
        max_y = shape[0]
        max_x = shape[1]
        p = {}
        grid = np.arange(ns).reshape(shape)
        it = np.nditer(grid, flags=["multi_index"])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            p[s] = {a: [] for a in range(na)}
            is_done = lambda s: s == 0 or s == (ns - 1)
            reward = 0.0 if is_done(s) else -1.0

            if is_done(s):
                p[s][UP] = [(1.0, s, reward, True)]
                p[s][RIGHT] = [(1.0, s, reward, True)]
                p[s][DOWN] = [(1.0, s, reward, True)]
                p[s][LEFT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if y == 0 else s - max_x
                ns_right = s if x == (max_x - 1) else s + 1
                ns_down = s if y == (max_y - 1) else s + max_x
                ns_left = s if x == 0 else s - 1
                p[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                p[s][RIGHT] = [(1.0, ns_up, reward, is_done(ns_right))]
                p[s][DOWN] = [(1.0, ns_up, reward, is_done(ns_down))]
                p[s][LEFT] = [(1.0, ns_up, reward, is_done(ns_left))]
            it.iternext()
        isd = np.ones(ns) / ns
        self.p = p
        super(GridWorldEnv, self).__init__(ns, na, p, isd)

    def _render(self, mode="human", close=False):
        if close:
            return

        # StringIO模块主要用于在内存缓冲区中读写数据
        outfile = StringIO() if mode == "ansi" else sys.stdout

        grid = np.arange(self.ns).reshape(self.shape)
        it = np.nditer(grid, flags=["multi_index"])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.ns - 1:
                output = "  T "
            else:
                output = " o "

            # lstrip()删除字符串开头指定的字符，默认为空格
            # rstrip()删除字符串末尾指定的字符，默认为空格
            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()