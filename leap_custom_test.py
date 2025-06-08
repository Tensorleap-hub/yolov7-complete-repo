from leap_binder import preprocess_func, input_encoder, gt_encoder, custom_loss, pred_visualizer, gt_visualizer
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('MacOSX')
from leap_binder import leap_binder
from code_loader.helpers import visualize
import onnxruntime as rt

def check_integration():
    check_generic = True
    plot_vis = True
    if check_generic:
        leap_binder.check()
    print("started custom tests")

    # load model
    model = tf.keras.models.load_model("yolov7-2.h5")
    # sess = rt.InferenceSession("yolov7.onnx")
    # input_name = sess.get_inputs()[0].name
    # label_name = sess.get_outputs()[-1].name

    res = preprocess_func()
    for set in res:
        for i in range(set.length):
            # get input and gt
            input = input_encoder(i, set)[None, ...]
            gt = gt_encoder(i, set)[None, ...]

            # infer model
            pred = model(input)
            # pred = sess.run([label_name], {input_name: np.moveaxis(inpt.astype(np.float32), [1, 2, 3], [2, 3, 1])})[0]

            # add batch
            batch_pred = tf.concat([pred, pred], axis=0)
            batch_gt = np.concatenate([gt, gt], axis=0)
            imgs = np.concatenate([input, input], axis=0)

            # metrics
            loss = custom_loss(batch_pred.numpy(), batch_gt, imgs)
            loss = custom_loss(pred.numpy(), gt, input)


            #vis
            img_with_bbox = pred_visualizer(pred.numpy(), input)
            gt_img_with_bbox = gt_visualizer(gt, input)

            if plot_vis:
                visualize(img_with_bbox)
                visualize(gt_img_with_bbox)


    print("Custom tests finished successfully")


if __name__ == '__main__':
    check_integration()
