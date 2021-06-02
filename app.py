from flask import Flask
from flask_restful import Resource, Api,reqparse
import glob
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from flask import render_template

app = Flask(__name__)
api = Api(app)

image_list = glob.glob(r'data/data/train_frames/train/*.png')

image_list.sort()



parser = reqparse.RequestParser()
parser.add_argument('integers', type=int, help='an integer for the accumulator')

@app.route('/', methods=['GET'])
def main():
    return render_template("index.html")
    

class Image(Resource):

    def get(self):
        args = parser.parse_args()
        id_image = args['integers']
        fname = image_list[id_image]
        img = image.load_img(fname)
        img = image.img_to_array(img)
        fig = plt.figure(figsize=(20,8))

        ax1 = fig.add_subplot(1,4,1)
        ax1.imshow(img)
        ax1.title.set_text('Actual frame')
        ax1.grid(b=None)
        # predict
        #results = predict(id_image)[0]
        # formatting the results as a JSON-serializable structure:
       
        return fname

api.add_resource(Image, '/image')



if __name__ == '__main__':
    
    app.run(debug=True)
    
    