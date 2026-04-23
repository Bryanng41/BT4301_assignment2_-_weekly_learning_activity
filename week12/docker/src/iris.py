from joblib import load



def classify(petallength, petalwidth):

    print('classify: ', petallength, petalwidth)

    dtree = load('./model/model.pkl')

    newX = [[petallength, petalwidth]]
    species = dtree.predict(newX)

    return {'species': species[0]}    
