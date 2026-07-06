import numpy as np
import pandas as pd
import tensorflow as tf 
import Maps

data = pd.read_csv("ob_data.csv")
TARGET = "NObeyesdad"
NUMERIC_COLUMNS = [
    "Weight",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE"
]
CATEGORICAL_COLUMNS = [
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS"
]
CATEGORY_MAP = {
    "Gender": {
        0: "Female",
        1: "Male"
    },
    "family_history_with_overweight": {
        0: "no",
        1: "yes"
    },
    "FAVC": {
        0: "no",
        1: "yes"
    },
    "CAEC": {
        0: "no",
        1: "Sometimes",
        2: "Frequently",
        3: "Always"
    },
    "SMOKE": {
        0: "no",
        1: "yes"
    },
    "SCC": {
        0: "no",
        1: "yes"
    },
    "CALC": {
        0: "no",
        1: "Sometimes",
        2: "Frequently",
        3: "Always"
    },
    "MTRANS": {
        0: "Walking",
        1: "Bike",
        2: "Motorbike",
        3: "Public_Transportation",
        4: "Automobile"
    }
}
normal_df = data[data[TARGET] == "Normal_Weight"]
normal_numeric = normal_df[NUMERIC_COLUMNS].mean()
normal_categorical = {
    col: normal_df[col].mode()[0]
    for col in CATEGORICAL_COLUMNS
}

def prepare_date():
    data = pd.read_csv("ob_data.csv")
    
    data['Gender'] = data['Gender'].map(Maps.gender_map)
    data['family_history_with_overweight'] = data['family_history_with_overweight'].map(Maps.yes_no_map)
    data['FAVC'] = data['FAVC'].map(Maps.yes_no_map)
    data['CAEC'] = data['CAEC'].map(Maps.caec_map)
    data['SMOKE'] = data['SMOKE'].map(Maps.yes_no_map)
    data['SCC'] = data['SCC'].map(Maps.yes_no_map)
    data['MTRANS'] = data['MTRANS'].map(Maps.mtrans_map)
    data['NObeyesdad'] = data['NObeyesdad'].map(Maps.obesity_map)
    data['CALC'] = data['CALC'].map(Maps.calc_map)
    train , validation , test = np.split(data.sample(frac=1) , [int(0.6*len(data)) , int(0.8*len(data))])
    train_x = train[:,:-1]
    train_y = train[:,-1]

    validation_x = validation[:,:-1]
    validation_y = validation[:,-1]

    test_x = test[:,:-1]
    test_y = test[:,-1]
    print(train_x.dtype)
    print(train_y.dtype)
    print(validation_x.dtype)
    print(validation_y.dtype)
    return train_x,train_y,validation_x,validation_y,test_x,test_y  

def train_model(train_x , train_y , validation_x , validation_y):
    nodes = 256 
    lr = 0.001
    epoch = 120
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(16,)))
    model.add(tf.keras.layers.Dense(nodes, activation='relu'))
    model.add(tf.keras.layers.Dense(nodes*2, activation='relu'))
    model.add(tf.keras.layers.Dense(nodes*2, activation='relu'))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_x , train_y , batch_size=32,epochs =epoch,validation_data=(validation_x,validation_y))
    return history,model


def plot(history):
    plt.plot(history.history["accuracy"] , label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    plt.plot(history.history["loss"] , label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

def predict(features,model):
    y_pred = model.predict(features)
    return {
        "prediction" : Maps.reverse_map[np.argmax(y_pred)],
        "probability" : y_pred[0].tolist()
    }

def suggestions(person):
    values = person.model_dump()
    recommendations = []
    for feature in NUMERIC_COLUMNS:
        current = values[feature]
        target = round(float(normal_numeric[feature]))
        if np.isclose(current, target, atol=0.05):
            continue

        recommendations.append({
            "feature": feature,
            "current": round(current, 2),
            "target": target,
            "direction": "increase" if current < target else "decrease",
            "difference": round(abs(current - target), 2)
        })
    
    for feature in CATEGORICAL_COLUMNS:
        current = round(values[feature])
        target = normal_categorical[feature]
        current_label = CATEGORY_MAP[feature].get(current, current)

        if current_label != target:
            if feature == "CALC" and current < Maps.calc_map[target]:
                continue
            recommendations.append({
                "feature": feature,
                "current": current_label,
                "target": target,
                "direction": "change"
            })
    return recommendations
