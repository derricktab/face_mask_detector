import tensorflow as tf

json_file = "model/model.json"
model_json = tf.keras.models.model_from_json(json_file)
model_json.save("kasujja.h5", save_format="hf")
#
# h5_file = "kasujja.h5"
# model_h5 = tf.keras.models.load_model(h5_file)
#
# print(model_h5.summary())
#
