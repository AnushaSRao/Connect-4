import tensorflow as tf

from tf_model import test_tf
from tf_model import train_tf
from tf_model import two_player_tf
from tf_model import ai_ai_tf

from keras_model import two_player_keras
from keras_model import train_keras
from keras_model import test_keras
from keras_model import ai_ai_keras
   
#########################################

while(1):
    print("********************CONNECT-4********************")
    print("Choose option:")
    print("1) Keras")
    print("2) Tensorflow")
    print("3) Exit")
    model_choice = int(input("Which model do you want to use? "))
    if(model_choice==2):
        while(1):
            print("Choose option:")
            print("1) Single Player")
            print("2) Two Player")
            print("3) AI vs AI")
            print("4) Train Model")
            print("5) Exit")
            choice = int(input("Enter choice :"))
            if(choice==5):
                break
            elif(choice==4):
                train_tf()
            elif(choice==3):
                saver = tf.train.Saver() 
                ai_ai_tf(saver)
            elif(choice==2):
                two_player_tf()
            else:
                saver = tf.train.Saver() 
                test_tf(saver)
            
    elif(model_choice==1):
        while(1):
            print("Choose option:")
            print("1) Single Player")
            print("2) Two Player")
            print("3) AI vs AI")
            print("4) Train Model")
            print("5) Exit")
            choice = int(input("Enter choice :"))
            if(choice==5):
                break
            elif(choice==4):
                train_keras()
            elif(choice==3):
                ai_ai_keras()
            elif(choice==2):
                two_player_keras()
            else:
                test_keras()
    else:
        break
    
#########################################