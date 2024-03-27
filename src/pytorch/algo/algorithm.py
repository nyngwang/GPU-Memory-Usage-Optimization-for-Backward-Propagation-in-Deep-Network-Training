import collections

from .scope import AlgoScope


# TODO: move to utils
def RANGE_CLOSED(l, r):
    return range(l, r+1) if l <= r else range(l, r-1, -1)


class Algorithm:
    def __init__(self, algoscope: AlgoScope):
        self.model = algoscope
        self.cmdargs = algoscope.cmdargs


    ## public methods.


    # compute segment memory for segment (l,r) where l,r are checkpoints.
    def segment_memory(self, l, r):
        if l > r: raise RuntimeError('In algorithm.py def segment_memory(l,r): got l > r.')
        # TODO: in algo3 we need to be aware of the def for those "empty segments".
        if l < r and l+1 == r:
            # WARN: if an empty segment has cost > 0, it will cause self.l to not allow ALL indexes in [1,n] and break algo2.
            return 0

        if self.cmdargs.smd == 'chen_et_al':
            return self.prefix_sum_cost_model[r-1] - self.prefix_sum_cost_model[l]
        elif self.cmdargs.smd == 'segment_cost_with_max':
            return (self.prefix_sum_cost_model[r-1] - self.prefix_sum_cost_model[l]) \
                   + self.range_max_cost_model[l, r-1][0]


    def solve(self, algo):

        if self.cmdargs.algo3:
            m, p, s, r = self.solve_dynamic_checkpoint_selection()

            p = p + self.model.cost_layer[-1] / 1024**2

            print('min:', m)
            print('segment memory:', s)
            print('checkpoint memory:', p)
            print('last checkpoint index of T(i):', r)

            # restore checkpoints.
            checkpoints = r.copy()
            print('FINAL checkpoints by algo3: {}'.format(checkpoints))
        else:
            m, p, s, r = self.solve_checkpoint_selection(algo) # O(n^3)

            p = p + self.model.cost_layer[-1] / 1024**2

            print('min:', m)
            print('segment memory:', s)
            print('checkpoint memory:', p)
            print('last checkpoint index of T(i):', r)

            # restore checkpoints.
            checkpoints = list()
            i = self.model.n

            # Collect all checkpoints. O(n)
            # NOTE: r[i] is the layer index of the right-most checkpoint on the sub-model [1, i].
            while r[i] != 0:
                # this checkpoint r[i] provides the right bound for the new sub-model [1, r[i]-1].
                checkpoints = [r[i]] + checkpoints
                i = r[i]-1

            # NOTE: checkpoints will not contain the last layer.
            # print('raw checkpoints: ', checkpoints)

        # NOTE: the last layer is a checkpoint.
        if checkpoints[-1] != self.model.n:
            checkpoints += [ self.model.n ]

        if self.cmdargs.cp:
            max_seg_mem = 0
            idx_seg_start = 0

            with open('./pytorch/ipynb/{}.txt'.format('checkpoints'), 'w') as file:
                checkpoints = self.cmdargs.cp.copy()
                if checkpoints[-1] != self.model.n:
                    checkpoints += [ self.model.n ]

                checkpoints_fw_bw = checkpoints + sorted([ (2*(self.model.n)+1-c) for c in checkpoints ])
                str_checkpoints = (','.join(map(str, checkpoints_fw_bw)))
                file.write(str_checkpoints)

            print('write to checkpoints.txt:', checkpoints_fw_bw)
            self.checkpoints = checkpoints
            return


        with open('./pytorch/ipynb/{}.txt'.format('checkpoints'), 'w') as file:
            checkpoints_fw_bw = checkpoints + sorted([ (2*(self.model.n)+1-c) for c in checkpoints ])
            str_checkpoints = (','.join(map(str, checkpoints_fw_bw)))
            file.write(str_checkpoints)

        print('write to checkpoints.txt:', checkpoints_fw_bw)
        self.checkpoints = checkpoints


    def solve_checkpoint_selection(self, f):
        ## prepare for the upcoming querys of {sum,max} in range [l, r].
        self.prepare_prebuild() # O(n^2)

        # NOTE: initialize with the all-checkpoint case
        checkpoint_memory = sum(self.model.cost_layer)
        segment_memory = max(self.model.cost_layer)
        min = checkpoint_memory + segment_memory
        print('min after initialization: ', min)

        print('DEBUG segment memory (0,3): ', self.segment_memory(0, 3))
        print('DEBUG segment memory (2,6): ', self.segment_memory(2, 6))
        print('DEBUG segment memory (3,6): ', self.segment_memory(3, 6))
        print('DEBUG segment memory (3,5): ', self.segment_memory(3, 5))
        print('DEBUG segment memory (2,4): ', self.segment_memory(2, 4))

        last_checkpoint = None

        # Enumerate segment-memory by end-points. O(n^2).
        # NOTE: ensure that d_0, d_n can be the left, right checkpoint, resp.
        for l in RANGE_CLOSED(0, self.model.n):
            for r in RANGE_CLOSED(l+1, self.model.n+1):
                max_seg_mem = self.segment_memory(l, r) # O(1)
                _min, _checkpoint_memory, _segment_memory = f(self.model.n, max_seg_mem) # O(n)
                if _min < min:
                    min = _min
                    checkpoint_memory = _checkpoint_memory
                    segment_memory = _segment_memory
                    last_checkpoint = self.model.idx_last_checkpoint.copy()

                self.__clear_cache()

        return min, checkpoint_memory, segment_memory, last_checkpoint


    ## draw prediction.


    def make_prediction(self, weight, algo2=False):

        model_len = self.model.n
        # print(self.checkpoints)
        print('>> model len: {}'.format(model_len))

        out = []

        wgm = round(weight*3, 2)

        # before forward, training phase index = 0.
        out += [ wgm ]

        # forward phase.
        for index_layer in RANGE_CLOSED(1, model_len):
            mem_layer = wgm

            # add the memory of both current and previous layers.
            mem_layer += self.model.cost_layer[index_layer-1]
            mem_layer += self.model.cost_layer[index_layer]

            # then accumulate those checkpoints before the previous layer.
            for idx_ckpt in self.checkpoints:
                if index_layer-1 >= idx_ckpt:
                    mem_layer += self.model.cost_layer[idx_ckpt]

            out += [ round(mem_layer, 2) ]

            # TODO: refactor
            if False or algo2:
                if index_layer in self.checkpoints:
                    mem_layer = wgm
                    for idx_ckpt in self.checkpoints:
                        if index_layer >= idx_ckpt:
                            mem_layer += self.model.cost_layer[idx_ckpt]
                    out += [ round(mem_layer, 2) ]


        # prepare for backward phase.

        mem_full_segment_with_buffer = 0
        mem_release_current_segment = 0
        index_current_segment_start = -1

        if algo2:
            # backward phase.
            for index_layer in RANGE_CLOSED(model_len, 1):
                # add memory of (weight + gradient-of-weight + momentum).
                mem_layer = wgm

                # add memory of all self.checkpoints.
                mem_layer += sum([ self.model.cost_layer[idx_ckpt] for idx_ckpt in self.checkpoints ])

                # reset the memory of output-gradient buffer upon the start of a new segment.
                if index_layer in self.checkpoints:
                    right = index_layer # we meet the right checkpoint first in the backward phase.
                    left = 0 if index_layer == self.checkpoints[0] else self.checkpoints[self.checkpoints.index(index_layer)-1]
                    print('left={}, right={}'.format(left, right))

                    mem_full_segment_with_buffer = self.segment_memory(left, right)
                    # print('mem_full_segment_with_buffer at backward layer-{}: {}'.format(index_layer, mem_full_segment_with_buffer))

                # add full memory of the current segment.
                mem_layer += mem_full_segment_with_buffer

                # at the beginning of each segment we haven't released any memory yet.
                if index_layer in self.checkpoints:
                    mem_release_current_segment = 0
                    index_current_segment_start = index_layer
                    if index_layer in self.checkpoints:
                        _mem_layer = wgm
                        _mem_layer += sum([ self.model.cost_layer[idx_ckpt] for idx_ckpt in self.checkpoints ])
                        out += [ round(_mem_layer, 2) ]
                else:
                    mem_release_current_segment += self.model.cost_layer[index_layer+1]

                    # NOTE: need to release the output-gradient buffer at the last layer within the current segment.
                    if index_layer-1 in self.checkpoints \
                        or index_layer == 1:
                        # print('index_current_segment_start: {}'.format(index_current_segment_start))

                        left = index_layer-1
                        right = index_current_segment_start-1
                        print('find max left={}, right ={}'.format(left, right))
                        mem_release_current_segment += max(self.model.cost_layer[left:right+1])

                # mem_layer -= mem_release_current_segment


                # print('backward layer-{}, mem_release_current_segment: {}'.format(index_layer, mem_release_current_segment))
                out += [ mem_layer ]
        else:
            # backward phase.
            for index_layer in RANGE_CLOSED(model_len, 1):
                # add memory of (weight + gradient-of-weight + momentum).
                mem_layer = wgm

                # add memory of all self.checkpoints.
                mem_layer += sum([ self.model.cost_layer[idx_ckpt] for idx_ckpt in self.checkpoints ])

                # reset the memory of output-gradient buffer upon the start of a new segment.
                if index_layer in self.checkpoints:
                    right = index_layer # we meet the right checkpoint first in the backward phase.
                    left = 0 if index_layer == self.checkpoints[0] else self.checkpoints[self.checkpoints.index(index_layer)-1]
                    print('left={}, right={}'.format(left, right))

                    mem_full_segment_with_buffer = self.segment_memory(left, right)
                    # print('mem_full_segment_with_buffer at backward layer-{}: {}'.format(index_layer, mem_full_segment_with_buffer))

                # add full memory of the current segment.
                mem_layer += mem_full_segment_with_buffer

                # index_layer-1 for algo3
                # at the beginning of each segment we haven't released any memory yet.
                if index_layer in self.checkpoints:
                    mem_release_current_segment = 0
                    index_current_segment_start = index_layer
                else:
                    # index_layer for algo3
                    mem_release_current_segment += self.model.cost_layer[index_layer+1]

                    # NOTE: need to release the output-gradient buffer at the last layer within the current segment.
                    if index_layer-1 in self.checkpoints \
                        or index_layer == 1:
                        # print('index_current_segment_start: {}'.format(index_current_segment_start))

                        left = index_layer-1
                        right = index_current_segment_start-1
                        print('find max left={}, right ={}'.format(left, right))
                        mem_release_current_segment += max(self.model.cost_layer[left:right+1])

                mem_layer -= mem_release_current_segment


                # print('backward layer-{}, mem_release_current_segment: {}'.format(index_layer, mem_release_current_segment))
                out += [ mem_layer ]

        # after backward, training phase index = 49.
        out += [ wgm ]

        # for record in out:
        #     print(record)
        # print('length: {}'.format(len(out)))

        return out


    ## private methods.


    def prepare_prebuild(self):
        # TODO: check clear cache.
        self.__prepare_prefix_sum() # O(n)
        self.__prepare_range_max_table() # O(n^2)


    def __prepare_prefix_sum(self):
        self.prefix_sum_cost_model = [0]
        for idx in RANGE_CLOSED(1, self.model.n):
            self.prefix_sum_cost_model.append(
                self.prefix_sum_cost_model[idx-1] + self.model.cost_layer[idx]
            )


    def __prepare_range_max_table(self):
        # NOTE: the range is [l,r], i.e. inclusive.
        self.range_max_cost_model = dict()

        for i in RANGE_CLOSED(0, self.model.n):
            self.range_max_cost_model[i, i] = (self.model.cost_layer[i], i)

        for l in RANGE_CLOSED(0, self.model.n-1):
            for r in RANGE_CLOSED(l+1, self.model.n):
                self.range_max_cost_model[l, r] = \
                    (self.model.cost_layer[r], r) \
                    if self.model.cost_layer[r] > self.range_max_cost_model[l, r-1][0] else \
                    self.range_max_cost_model[l, r-1]


    def __clear_cache(self):
        model_len = self.model.n
        self.model.t_cache = [self.model.INF] * (model_len+1)
        self.model.idx_last_checkpoint = [0] * (model_len+1)


    # algorithms.


    # Method 1,2: BF Search w(/o) cut.
    def BF(self, model_len, max_seg_mem, cut=True):
        min = self.model.INF
        seg_mem = 0
        checkpoint_mem = 0
        counter_top_down = 2**model_len - 1
        
        for i in RANGE_CLOSED(counter_top_down, 0):
            # in the binary, 1=select 0=no-select.
            _checkpoint_mem = 0
            _max_seg_mem = 0

            _seg_mem = 0
            _max_cost_layer = 0
            should_cut = False

            # check every bit of i, it's a checkpoint if it is 1.
            for j in RANGE_CLOSED(model_len-1, 0):
                bit_tractor = 2**j
                cost_current_layer = self.model.cost_layer[model_len-j]

                # is a checkpoint.
                if i & bit_tractor:
                    _seg_mem = _seg_mem + _max_cost_layer # double the cost of the max layer.
                    if _seg_mem > _max_seg_mem:
                        _max_seg_mem = _seg_mem
                    _seg_mem = 0
                    _max_cost_layer = 0
                    _checkpoint_mem += cost_current_layer
                else:
                    if cost_current_layer > _max_cost_layer:
                        _max_cost_layer = cost_current_layer

                    _seg_mem += cost_current_layer

                    # early stop.
                    if cut and (_seg_mem+_max_cost_layer) > max_seg_mem:
                        should_cut = True
                        break

            if cut and should_cut:
                continue

            # no early stop, we need to fix a situation that the last layer is not checkpoint.
            if _seg_mem != 0:
                _seg_mem = _seg_mem + _max_cost_layer
                if _seg_mem > _max_seg_mem:
                    _max_seg_mem = _seg_mem

            # update if the sum is better.
            if (_checkpoint_mem + _max_seg_mem) < min:
                min = (_checkpoint_mem + _max_seg_mem)
                checkpoint_mem = _checkpoint_mem
                seg_mem = _max_seg_mem

        return min, checkpoint_mem, seg_mem


    # Method 5: DP bottom-up with mono-queue, O(n).
    def T_bottom_up_mono(self, model_len, max_seg_mem):

        self.model.t_cache[0] = 0

        # the 0 means no-checkpoint is selected.
        # the only way to pop-out 0 is by the constrant of `max_seg_mem`, since it's zero-cost.
        q = collections.deque([0])

        self.prebuild_l(max_seg_mem)

        # bottom-up, O(n)
        for i in RANGE_CLOSED(1, model_len):
            while len(q) > 0 \
                and q[-1] > 0 \
                and self.model.t_cache[i-1] + self.model.cost_layer[i] \
                    < self.model.t_cache[q[-1]-1] + self.model.cost_layer[q[-1]]:
                # 1. when both exist in the q: the value of the former is better.
                # 2. the former will live longer since the cost of its right-most segment is 0.
                q.pop()
            q.append(i)

            while q[0] < self.l[i]:
                q.popleft()

            if q[0] == 0:
                self.model.t_cache[i] = 0
            else:
                self.model.t_cache[i] = self.model.t_cache[q[0]-1] + self.model.cost_layer[q[0]]

            self.model.idx_last_checkpoint[i] = q[0]

        checkpoint_memory = self.model.t_cache[model_len]

        # compute segment memory, O(n)
        checkpoints = []
        idx_last_checkpoint = self.model.idx_last_checkpoint[model_len]
        # idx_last_checkpoint 同時也是 model length.
        while idx_last_checkpoint > 0:
            checkpoints = [idx_last_checkpoint] + checkpoints
            idx_last_checkpoint = self.model.idx_last_checkpoint[idx_last_checkpoint-1]
        i = 0
        segment_memory = 0
        for j in checkpoints:
            _segment_memory = self.segment_memory(i, j) # O(1)
            if _segment_memory > segment_memory:
                segment_memory = _segment_memory
            i = j
        # the last segment, i is now the last checkpoint.
        _segment_memory = self.segment_memory(i, model_len+1)
        if _segment_memory > segment_memory:
            segment_memory = _segment_memory

        min = checkpoint_memory + segment_memory

        return min, checkpoint_memory, segment_memory


    # prebuild l(i, s), O(n).
    # NOTE: return the leftmost possible j index in the objective function.
    # method: sliding window.
    def prebuild_l(self, max_seg_mem):
        self.l = [0] # when the model is of length 0.
        j = 0 # index of the last checkpoint.
        for model_len in RANGE_CLOSED(1, self.model.n):
            # because we assume that the "imaginary model_len+1 layer" is a checkpoint.
            while j<=model_len and self.segment_memory(j, model_len+1) > max_seg_mem:
                j += 1

            self.l.append(j)

        # length of self.l becomes 1+model_len.


    # Method 3: DP top-down w/ cache.
    def T_top_down(self, i, max_seg_mem):
        if self.model.t_cache[i] != self.model.INF:
            return self.model.t_cache[i]
        if i == 0:
            return 0
        # if i == 1:
        #     return 0 if CONTEXT['LAYER_DATA'][0] <= s else CONTEXT['LAYER_DATA'][0]

        min = self.model.INF

        for j in RANGE_CLOSED(self.l[i], i):
            c_j = self.model.cost_layer[j-1] if j >= 1 else 0

            # REF: formula (3)
            _min = c_j + self.T_top_down(j-1, max_seg_mem)

            if _min < min:
                min = _min


        self.model.t_cache[i] = min
        return min


    # Method 4: DP bottom-up w/ cache.
    def T_bottom_up(self, i, s):
        # start case
        self.model.t_cache[0] = 0

        for _i in RANGE_CLOSED(1, i):
            min = self.model.INF
            for j in RANGE_CLOSED(self.l[_i], _i):
                c_j = self.model.cost_layer[j-1] if j >= 1 else 0
                _min = c_j + (self.model.t_cache[j-1] if j >= 1 else 0)

                if _min < min:
                    min = _min

            self.model.t_cache[_i] = min

        return self.model.t_cache[i]


    # for algorithm 3.


    # TODO: add a parameter to toggle debug-logging.
    # FORMULA: U(i,j) = s(i,j) + d_j.
    # NOTE: U(i,j) means that the maximum occurs at recomputing the segment (i,j).
    def U(self, i, j):
        s_ij = self.segment_memory(i, j)
        # print('>>> s({},{}): {}'.format(i, j, s_ij))

        return s_ij + self.model.cost_layer[j]


    # FORMULA: T(i) = d_i + min_{i<j<n}(max( U(i,j), T(j) )).
    # * d_i is a checkpoint.
    # * consider model range [i, n] only.
    # * just finish step i+1, so the recomputing has NOT happen. (this is why T(j) works)
    def T_algo3(self, i):
        # print('\n-----------> i = {}'.format(i))
        # print('cost_layer: {}'.format(self.model.cost_layer[1:]))

        min_without_di = self.model.INF
        idx_j_min = -1

        # enumerate i < j <= n, where j=n means that we cannot find j in range i < j < n.
        for j in range(self.model.n, i, -1):
            _Uij = self.U(i, j)
            _max = max(_Uij, self.model.t_cache[j])
            # when i, j are adjacent, add memory of i into U(i,j).
            if i+1 == j:
                _max = max(_max, _Uij+self.model.cost_layer[i])
            # print('>> j={}, U(i,j)={}, T(j)={}, _max={}'.format(j, _Uij, self.model.t_cache[j], _max))

            if _max < min_without_di:
                min_without_di = _max
                idx_j_min = j


        self.model.t_cache[i] = self.model.cost_layer[i] + min_without_di
        # print('>> T({})={}'.format(i, self.model.t_cache[i]))

        self.model.idx_last_checkpoint[i] = idx_j_min
        return self.model.t_cache[i]


    def solve_dynamic_checkpoint_selection(self):
        self.__clear_cache()
        self.prepare_prebuild() # O(n^2)

        # NOTE: T(n) is d_n*2 (includes the r-buffer) for now.
        self.model.t_cache[self.model.n] = self.model.cost_layer[self.model.n]*2
        self.model.idx_last_checkpoint[self.model.n] = self.model.n

        all_T = []
        # O(n^2)
        # NOTE: the answer is at T(0) so that:
        # * d_0, d_n are checkpoints.
        # * and d_1 might not be a checkpoint.
        for i in range(self.model.n-1, -1, -1):
            # O(n)
            all_T = [ self.T_algo3(i) ] + all_T
        all_T += [ self.model.t_cache[self.model.n] ]
        print('T[0 ~ n-1]: {}\n'.format(all_T))

        ############## restore checkpoints, O(n).

        # NOTE: each j in T(i) is the "first" checkpoint after i.
        idx_first_checkpoint = self.model.idx_last_checkpoint.copy()
        print('idx_first_checkpoint: {}'.format(idx_first_checkpoint))
        checkpoints = []
        idx_i = 0
        while idx_i < self.model.n:
            j_min = idx_first_checkpoint[idx_i]
            checkpoints = checkpoints + [ j_min ]
            idx_i = j_min
        print('checkpoints: {}\n'.format(checkpoints))

        # checkpoint memory. O(n)
        checkpoint_memory = 0
        for idx_ckpt in checkpoints:
            checkpoint_memory += self.model.cost_layer[idx_ckpt]
        print('checkpoint memoery:', checkpoint_memory)

        # largest segment memory. O(n)
        # TODO: need to take care of adjacent checkpoints.
        i = 0
        segment_memory = -1
        for j in checkpoints[:-1]:
            _segment_memory = self.segment_memory(i, j) # O(1)
            if _segment_memory > segment_memory:
                segment_memory = _segment_memory
            i = j
        # the last segment, i is now the last checkpoint.
        _segment_memory = self.segment_memory(i, self.model.n)
        if _segment_memory > segment_memory:
            segment_memory = _segment_memory

        min = checkpoint_memory + segment_memory

        return min, checkpoint_memory, segment_memory, checkpoints

