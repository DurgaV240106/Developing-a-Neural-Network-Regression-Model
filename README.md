# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Neural Network Regression is used to predict continuous numeric values from input data. In this experiment, a neural network model is developed using one input feature and one output value to learn the relationship between them. The network consists of input, hidden, and output layers, where hidden layers use activation functions to capture patterns in the data. The model is trained using training data by minimizing Mean Squared Error (MSE) with an optimizer. After training, the model can accurately predict output values for new unseen inputs.

## Neural Network Model
<img width="431" height="296" alt="image" src="https://github.com/user-attachments/assets/43c2a349-b3ad-47a7-8890-fe838fe01452" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:DURGA V

### Register Number:212223230052
```

class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1, 8)
        self.fc2=nn.Linear(8, 10)
        self.fc3=nn.Linear(10, 1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
  def forward(self, x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer

lig=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(lig. parameters(), lr=0.001);



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range (epochs):
        optimizer. zero_grad()
        loss=criterion(ai_brain(X_train), y_train)
        loss. backward()
        optimizer.step()
        lig .history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```

## Dataset Information:

<img width="111" height="128" alt="image" src="https://github.com/user-attachments/assets/53a9d074-0169-4859-8cdd-ab1da646eb79" />

## OUTPUT :

<img width="215" height="306" alt="image" src="https://github.com/user-attachments/assets/49d3e01a-e34c-4c42-b2f2-8d10c1283e77" />

## Training Loss Vs Iteration Plot:
<img width="452" height="329" alt="image" src="https://github.com/user-attachments/assets/6d587d86-c327-46a3-b4b7-90a22c75bdca" />
## New Sample Data Prediction:
Prediction: 32.45441818237305
## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
