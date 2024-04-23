# Importar bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble

class Predictor():
    def __init__(self, csv_file="data/kc_house_data.csv") -> None:
        # Cargar datos
        self.data = pd.read_csv(csv_file)
        
        # Definimos las etiquetas
        labels = self.data['price']

        # Creamos un DataFrame de entrenamiento removiendo
        # las columnas id y price
        self.train = self.data.drop(["id", "price"], axis=1)

        # Generamos conjuntos de entrenamiento y de prueba
        x_train , x_test , y_train , y_test = train_test_split(self.train , labels , test_size = 0.10,random_state =2)

        # Generamos un regresor utilizando ensamblajes y Gradient Bootsting
        self.regressor = ensemble.GradientBoostingRegressor(n_estimators = 400,
                                                            max_depth = 5,
                                                            min_samples_split = 2,
                                                            learning_rate = 0.1,
                                                            loss = 'ls')
        
        # Entrenamos el regresor
        self.regressor.fit(x_train, y_train)

    def predict(self, input):
        self.prediction = self.regressor.predict(input)
        return self.prediction




