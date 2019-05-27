import * as tf from '@tensorflow/tfjs';

/**
 * A dataset for webcam controls which allows the user to add example Tensors
 * for particular labels. This object will concat them into two large xs and ys.
 */
export class ControllerDataset {

    constructor(numClasses) {
        this.numClasses = numClasses;
    }
    /**
     * Adds an example to the controller dataset.
     * @param {Tensor} example A tensor representing the example. It can be an image,
     *     an activation, or any other type of Tensor.
     * @param {number} label The label of the example. Should be a number.
    */
    addExample(example, label) {
        // One-hot encode the label.
        const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));

        if (this.xs == null) {
            // For the first example that gets added, keep example and y so that the
            // ControllerDataset owns the memory of the inputs. This makes sure that
            // if addExample() is called in a tf.tidy(), these Tensors will not get
            // disposed.
            this.xs = tf.keep(example);
            this.ys = tf.keep(y);

        } else {
            
            // all other examples
            const oldX = this.xs;
            const oldY = this.ys;

            // concatenate previous and new examples and keep for later
            this.xs = tf.keep(oldX.concat(example, 0));
            this.ys = tf.keep(oldY.concat(y, 0));
            
            // specifcally call dispose instead of tidying
            // lots of tensors mean lots of memory, js memory management isnt great which is why tensor flow opts to handle alot of gabbage collection explicitly
            oldX.dispose();
            oldY.dispose();
            y.dispose();
        }
    }
}
