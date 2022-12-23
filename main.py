from functions.githubSeg import cloth_color
from functions.compare_colors import compare_colors
import sys
import pickle



if __name__ == "__main__":
    loaded_model = pickle.load(open('knnpickle_file', 'rb'))
    res = cloth_color(sys.argv[1:][0])
    for each in res:
        each[0].reverse()
        result = loaded_model.predict([each[0]])
        final_colour = compare_colors(result[0])
        print(final_colour, each[1])
