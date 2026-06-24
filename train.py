import tensorflow as tf

# Load dataset
train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "flowers",
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),

    # 5 flower classes
    tf.keras.layers.Dense(5)
])

# Compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
model.save("flower_model.h5")

print("Model saved successfully!")