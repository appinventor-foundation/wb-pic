# WB PIC
An offline version of the Personal Image Classification website, to be used with MIT App Inventor's personal image classifier extension or as a standalone website.

## Folder Structure
The `mobilenet` folder contains shards and model.json file for v1 of mobilenet.
The `mobilenetv2` folder contains shards and model.json file for v2 of mobilenet.
The `newpic` folder contains the sources for the React app to perform training and testing for the Personal Image Classifier website.
The server version of the training functionality is in the `newpic/server` folder and it contains its own npm package


## Architecture
### Front end
A React app is the main building block of the UI for this site.


### Server mode
An optional nodejs, expresjs app can be run to take on higher loads of training data.


PIC has two training modes, directly on the browser or delegated to a server.

The server will load a different package depending on if it can detect a GPU or not. If it is detected, the GPU will be used for training and if not, it will default to CPU.

## Development
Both the React app and the server have their own set of dependencies and npm files.

Each of these sets must be installed with:
```bash
npm install
```

Each of them can be started for development (hot reload) with:
```bash
npm start
```


## Deployment
There are a number of ways that this software can be deployed.

### Standalone, in browser only app
This mode can be deployed by simply serving the files from the output of `npm build` on any web server such as Apache or Nginx.

### Browser and Server training
This mode needs nodejs installed in the server machine. The sources in the `newpic/server` folder can be run with the node command; it is an expressjs app.
Running in server mode also means that the React app or front end could be served as static assets from the same node server.

For production the recommended setup would be nginx as reverse proxy serving the app as statis aseets and the node app in a local port. Docker images can also be generated.


## App Specifics - Training Settings

The app also allows to load models from a URL if connectivity is available.
