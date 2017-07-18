$(document).ready(function () {
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port);
    console.log('connected http://' + document.domain + ':' + location.port);
    console.log(socket);
    var numbers_received = [];

    //receive details from server
    socket.on('new_image', function (msg) {
        console.log("Received image");
        result_obj = $.parseJSON(msg.classification);
        $('#result').html('');

        $('#img').attr('src', 'data:image/png;base64,' + msg.image);

        for (pred in result_obj['predictions']) {
            console.log("blablabla2");
            $('#result').append("Prediction: " + pred['highest']);
        }

    });

})
;