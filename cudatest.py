import tensorflow as tf

print("TensorFlow versiyonu:", tf.__version__)
print("GPU destekli mi:", tf.test.is_built_with_cuda())
print("GPU erişimi var mı:", tf.config.list_physical_devices('GPU'))
