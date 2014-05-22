
from theano import tensor
from theano.tensor.signal.downsample import DownsampleFactorMax


def max_pool_3d(input, ds, ignore_border=False):
    """
    Takes as input a N-D tensor, where N >= 3. It downscales the input by
    the specified factor, by keeping only the maximum value of non-overlapping
    patches of size (ds[0],ds[1],ds[2]) (depth, height, width)

    Arguments:
        input (N-D theano tensor of input images): input images. Max pooling will be done over the 3 last dimensions.
        ds (tuple of length 3): factor by which to downscale. (2,2,2) will halve the video in each dimension.
        ignore_border (boolean): When True, (5,5,5) input with ds=(2,2,2)
            will generate a (2,2,2) output. (3,3,3) otherwise.
    """

    if input.ndim < 3:
        raise NotImplementedError('max_pool_3d requires a dimension >= 3')

    # extract nr dimensions
    vid_dim = input.ndim
    # max pool in two different steps, so we can use the 2d implementation of
    # downsamplefactormax. First maxpool frames as usual.
    # Then maxpool the depth dimension. Shift the depth dimension to the third
    # position, so rows and cols are in the back

    # extract dimensions
    frame_shape = input.shape[-2:]

    # count the number of "leading" dimensions, store as dmatrix
    batch_size = tensor.prod(input.shape[:-2])
    batch_size = tensor.shape_padright(batch_size,1)

    # store as 4D tensor with shape: (batch_size,1,height,width)
    new_shape = tensor.cast(tensor.join(0, batch_size,
                                        tensor.as_tensor([1,]),
                                        frame_shape), 'int32')
    input_4D = tensor.reshape(input, new_shape, ndim=4)

    # downsample mini-batch of videos in rows and cols
    op = DownsampleFactorMax((ds[1],ds[2]), ignore_border)
    output = op(input_4D)
    # restore to original shape
    outshape = tensor.join(0, input.shape[:-2], output.shape[-2:])
    out = tensor.reshape(output, outshape, ndim=input.ndim)

    # now maxpool depth

    # output (depth, rows, cols), reshape so that depth is in the back
    shufl = (list(range(vid_dim-3)) + [vid_dim-2]+[vid_dim-1]+[vid_dim-3])
    input_depth = out.dimshuffle(shufl)
    # reset dimensions
    vid_shape = input_depth.shape[-2:]

    # count the number of "leading" dimensions, store as dmatrix
    batch_size = tensor.prod(input_depth.shape[:-2])
    batch_size = tensor.shape_padright(batch_size,1)

    # store as 4D tensor with shape: (batch_size,1,width,depth)
    new_shape = tensor.cast(tensor.join(0, batch_size,
                                        tensor.as_tensor([1,]),
                                        vid_shape), 'int32')
    input_4D_depth = tensor.reshape(input_depth, new_shape, ndim=4)
    # downsample mini-batch of videos in depth
    op = DownsampleFactorMax((1,ds[0]), ignore_border)
    outdepth = op(input_4D_depth)
    # output
    # restore to original shape (xxx, rows, cols, depth)
    outshape = tensor.join(0, input_depth.shape[:-2], outdepth.shape[-2:])
    shufl = (list(range(vid_dim-3)) + [vid_dim-1]+[vid_dim-3]+[vid_dim-2])
    return tensor.reshape(outdepth, outshape, ndim=input.ndim).dimshuffle(shufl)