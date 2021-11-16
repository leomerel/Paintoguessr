import cnn
import plot
import paint

if __name__ == '__main__':
    cnn = cnn.Cnn()
    #model = cnn.load_model("2021-11-16_16-24-23(80%)")
    model = cnn.load_model()
    predictions = cnn.test_model(model)

    paint.Paint(cnn)

