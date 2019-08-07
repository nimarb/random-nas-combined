# Notes

## Experiment notes

* experiments until 2019-07-24 were run with `--num_min_depth=30` and `--num_max_depth=150`.
* experiments after 2019-07-24-18-00 were run with `--num_min_depth=20` and `--num_max_depth=70`. The reason being that due to the vanishing gradient VGG will not train for deep nets

## Observations

* Training vgg takes at least an order of magnitude less time than training
  resnet based networks.
  * This statement was **very incorrect**, as the short training time was caused
    by a bug in the code. Testing at the end of training failed due to incorrect
    paths.

## Features / implementations

### DenseNet blocks

#### Similarities and differences DenseNet vs ResNet

Similar to ResNet, DenseNet adds shortcuts among layers. Different from Resnet,
a layer in dense receives all the outs of previous layers and concatenate them
in the depth dimension. In Resnet, a layer only receives outputs from the
previous second or third layer, and the outputs are added together on the same
depth, therefore it won’t change the depth by adding shortcuts. In other words,
in Resnet the output of layer of k is x[k] = f(w * x[k-1] + x[k-2]), while in
DenseNet it is x[k] = f(w * H(x[k-1], x[k-2], … x[1])) where H means stacking
over the depth dimension. Besides, Resnet makes learn identity function easy,
while DenseNet directly adds identity function.
Source: [here](https://medium.com/@smallfishbigsea/densenet-2b0889854a92)

#### Architecture

Within a DenseBlock, each layer is directly connected to every other layer in
front of it (feed-forward only).

The **Block config** looks like this: `(6, 12, 24, 16)` and indicates how many
layers each pooling block contains.

##### Current bug in SepConv:

```python
RuntimeError: Given groups=32, weight of size 32 1 5 5, expected input[128, 16, 16, 16] to have 32 channels, but got 16 channels instead
```

##### Current bug in `checkpoint(bn_function)` in `_DenseLayer`

```python
RuntimeError: running_mean should contain 16 elements not 64
```

The actual error here is, that the output of The conv layer in the
`_Transition` layer has to match the input of the next `norm1` layer.

#### PyTorch implementations

* [official pytorch implementation](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py)
  * Gives some errors, see above
* [very clean implementation](https://github.com/kevinzakka/densenet)
  * seems to have not ideal performance according to the author

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

Attention: **THE FIX DESCRIBED ABOVE IS INCORRECT**

The **correct fix** is, by making each layer directly connected to its previous
layer and not randomising the connection. Then, we randomise the number `num_depth`
at the beginning of execution.

### Number of ResNet layers not random

The number of layers for the ResNet structure is not random even though the `num_depth`
variable is being randomised.

#### Infos/observations of the bug

* 

#### Tried fixes

#### Fix
