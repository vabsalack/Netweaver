## Areas of Enhancing
1. In Model's training method - create a log file for logging the training progress details and make a graph out of it. Try real time graph updates or for every checkpoints
2. In Model's training method - shuffle the entire data for every epoch
3. Help linter: find a way out to layer.next.dinputs and layer.prev.dinputs. create   empty next and prev variable in all layer in model layer list.



## End goal
1. Understand and Leanring
    1. learn 5H1W frameworks of?
        1. Dense Layer
        2. Various Activations functions
        3. Loss Functions
        4. Forward pass
        6. Backward pass, chain rule, dweights, dbiases, dinputs
        7. Optimizers, and it's tuning parameters
        8. types of Regularization (Dropout, L1 and L2)

2. drop your ideas to implementation?
    1. clear your doubts and ideate and implement at ease.
    2. Go behind the scenes of neural network play. take control and alter the play.

3. this project gathers collabrators and learners, tends to form an community
    1. Create common community platform for learners and collabrators.
    2. I chose discord as community platform

4. Project release
    1. Existing release
        1. bug fix.
    2. Upcoming release
        1. set number of features to accumulate for a release

## Plan of action for end goal
1. Develop the library
    1. develop: clearly write the develop files, comments and package doc or purpose
    2. test it: clearly write fixtures, unit, integration test, functional test, e2e
    3. doc the test
    4. build into package, add travis CI to github
    5. create a proper readme file
        1. steps to install the package.
        2. steps to recreate the developing enviroment using uv
        3. Instructions to create Pull Requests and issues on github
    6. 



## using docker for development
1. ### goal
    1. Only prerequestie is docker, vscode with dev container extension
        1. docker file -> image(debian-full, uv's python3.13.3)
            1. create the user
            2. download uv installer script
            3. install uv in user .local bin
            4. install uv managed python version 3.13.3
        2. devcontainer.json
            1. mount {project root} inside /home/user/proejct-netweaver/
            2. execute uv lock, uv lock upgrade, uv sync (install project also)
            3. uses host project folder to create .venv and .cache/uv-cache/

# todo 21 may
1. perfect the doc of classes and functions.
2. make readme handy
3. make contributing handy

## readme
1. cool/creepy netweaver image.
2. Table of contents
2. cool introduction
    1. what is the library?
        1. Netweaver, helps us to weave the neural net from scratch. see it as a weaver who doesn't misbehave your commands. Personally me and I hope, lot's of you, out somewhere, popped out with questions and crazy ideas during learning process of deep learning, left helpless because of don't know where to start. The netweaver gives you a arena to transform you ideas to something really works and useful. 
    2. why is it is built?
        1. It is built for learning and eductional purpose. some popular deep learning library has many layers of abstraction, which make the code break or clueless of where to start to change. The Netweaver, I made the abstraction less as possible. so, you can implement your ideas without trouble.
    3. how it can be benefit the user?
        1. I introduced additional functionality such as live plotting within the notebook. all you need to do is to do tweaks and adjust the code accord to yoru needs. If user need to add additional functionality such as new type of layer like CNN, activation functions like softplus.., or new optimizer. kindly fork the repo and PR your changes. contributions are heartly welcome.


## add features
1. bbox for latset metrics following
2. animate the weights params
3. add conv layer
4. add new loss functions
5. 

## readme modification


## git commands
branch
git push origin -u branch_name # push the new branch, -u flags configure new_branch to follow remote tracking br.
git tag tag_name # create new tag on current commit
git push origin --tags # push the tags to remote origin
uv version --bump major/minor/patch

##links
shield.io - https://shields.io/
simpleicons - https://simpleicons.org/
color select - https://htmlcolorcodes.com/color-picker/
daily.dev - readme badge article


v1.0.2 changes
1. fix _load_data from dataset.py
    1. mlp.img scales image to [0, 1] when reading, which conflicts with manual scaling after
    2. fix = replace mpl.img with PIL.Image.Open then image = np.array(img)
2. add extra summary lines
    1. add details about the distinct labels of train and test sets in output of summary function in datasets.py

