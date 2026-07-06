import ObModel as om


train_x , train_y , validation_x, validation_y , test_x , test_y  = om.prepare_date()
history , NN_model = om.train_model(train_x=train_x ,train_y=train_y , validation_x=validation_x , validation_y=validation_y)
NN_model.save("obesity_model.keras")