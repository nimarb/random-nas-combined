# Notes

## Bugs

### Number of layers hovers around 30

#### Infos on the bug

* The number of `True` in `self.is_active` determines the number of layers in
  the network. It is in `cgp.py`.
* The `self.is_active` elements are set to `True` in the recursive function
  `__check_course_to_out(self, n)`.
  * It traverses the `self.gene` array in which each element has a structure
    like this: `[layer_type nr][rnd val 0..index]`. The number of elements is
    equal to `node_num + out_num`.
  * `self.gene` is populated in `init_gene(self)`.
  * The recursive stop condition is with `input_num = 1` and the first `n` the
    function is called with is `201` or the max node number incl. output nodes.
    At this time, all `is_active` elements are also `False`.

    ```python
    if not self.is_active[n]:
        ...
        for i in range(in_num):
            if self.gene[n][i+1] >= self.net_info.input_num:
                self.__check_course_to_out(
                    self.gene[n][i+1] - self.net_info.input_num)
    ```

* The actual layer number is defined in `active_net_list()` .

* The number of layers is actually determined by the randomly successfull
  connections made in the `__check_course_to_out(self, n)` function. There,
  possible viable paths are checked. Then afterwards, it is checked if the
  number of selected layers is more than the specified `num_min_depth`. If not,
  `mutation(self, mutation_rate)` is called to create a valid mutation with
  more layers.

#### Tried fixes

* The random generator is not the problem, for the random value `0..index` the
  number `1` appears 10x more than other values but this is expected. For the
  layer type, type `7` appears `1.5..2` times as often as any other layer type.
* Increasing the number of `num_min_depth` to a higher number than the usual
  number of layers. However, this kind of results in an endless loop as the
  computer needs a long time to come up with random numbers which result in a
  connection connecting enough layers together.

#### Fix

The logic of connecting layers has to be rewritten, this impacts `init_gene()`
and `__check_course_to_out(self, n)`.

Solution is in the `level_back` variable which indicates the distance between
the selected layers. `level_back=1` means each layer is connected to its following
one.
