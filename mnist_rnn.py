import tensorflow as tf
from tensorflow.contrib import rnn


#import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)

#Define constants
time_steps=28
n_hidden=128
n_input=28
learning_rate=0.001
n_classes=10
batch_size=128


#define weights and bias between output of hidden to final mapping
out_weights=tf.Variable(tf.random_normal([n_hidden,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#defining placeholders
x=tf.placeholder("float",[None,time_steps,n_input])
y=tf.placeholder("float",[None,n_classes])

#processing the input tensot from [batch_size,n_steps,n_input] to n_steps number of [batch_size,n_input] tensors
input=tf.unstack(x ,time_steps,1)



#defining the network
lstm_layer=rnn.BasicLSTMCell(n_hidden,forget_bias=1)


outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")


#converting last output of dimension [batch_size,n_hidden] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.matmul(outputs[-1],out_weights)+out_bias

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#initialize variables
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    iter=1

    while iter<800:
        batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)

        batch_x=batch_x.reshape((batch_size,time_steps,n_input))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iter %10==0:

            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})


            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")

        iter=iter+1



#calculating test accuracy
    test_data = mnist.test.images[:128].reshape((-1, time_steps, n_input))
    test_label = mnist.test.labels[:128]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

'''''

image,label=mnist.train.next_batch(3)
image=image.reshape((3,28,28))



image=tf.unstack(image,28,1)
sess=tf.InteractiveSession()
init = tf.global_variables_initializer()
img=sess.run(image)
print(type(img))
print(img)
print(img[-1])


'''''

'''''
image=mnist.train.images[10]
pixels = image.reshape((28, 28))
plt.imshow(255*pixels,cmap=plt.get_cmap('gray_r'))
plt.show()

'''''
