$(document).ready(function () {
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port);
    console.log('connected http://' + document.domain + ':' + location.port);
    console.log(socket);
    var numbers_received = [];

    //receive details from server
    socket.on('new_image', function (msg) {
        console.log("Received image");

        $('#img').attr('src', 'data:image/png;base64,' + msg.image);
        /*
         //maintain a list of ten numbers
         if (numbers_received.length >= 10){
         numbers_received.shift()
         }
         numbers_received.push(msg.number);
         numbers_string = '';
         for (var i = 0; i < numbers_received.length; i++){
         numbers_string = numbers_string + '<p>' + numbers_received[i].toString() + '</p>';
         }
         $('#log').html(numbers_string);
         */
    });

});