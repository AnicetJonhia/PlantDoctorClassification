
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

dataset_dir = "datasets"
img_size = (224, 224)
batch_size = 8

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    dataset_dir, target_size=img_size, batch_size=batch_size, subset="training"
)
val_gen = datagen.flow_from_directory(
    dataset_dir, target_size=img_size, batch_size=batch_size, subset="validation"
)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*img_size,3))
x = GlobalAveragePooling2D()(base.output)
predictions = Dense(len(train_gen.class_indices), activation="softmax")(x)
model = Model(inputs=base.input, outputs=predictions)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_gen, validation_data=val_gen, epochs=3)
model.save("plant_model.h5")
print("Modèle enregistré : plant_model.h5")
