$(document).ready(function () {
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port);
    console.log('connected http://' + document.domain + ':' + location.port);
    //console.log(socket);
    var classification_div = $('#classifications');

    //receive details from server
    socket.on('new_image', function (msg) {
        console.log("Received image");
        result_obj = $.parseJSON(msg);
        var img_src = 'data:image/jpg;base64,' + result_obj['image'];
        var image_id = result_obj['image_id'];
        image_id_text = '<p>Image ID: ' + result_obj['image_id'] + '</p>';
        var img_text = '<div class="col-md-8"><img id=' + image_id + ' src=' + img_src + ' class="img-rounded recent"></div>';
        var html_result = '<div class="row margin_top well">' + img_text;
        html_result += '<div class="col-md-4">';
        if (result_obj['classification']['predictions'].length === 0) {
            html_result += '<div class="col-md-6"><div id="result">' + image_id_text + ' ' +
                '<p>No Faces found</p></div></div>';
        }
        else {
            for (var i = 0; i < result_obj['classification']['predictions'].length; i++) {
                var probabilities_text = '<p>';

                var res = result_obj['classification']['predictions'][i];
                var probabilities = res['probabilities'];

                // Get an array of the keys:
                var keys = Object.keys(res['probabilities']);

                // Then sort by using the keys to lookup the values in the original object:
                keys.sort(function (a, b) {
                    return res['probabilities'][b] - res['probabilities'][a];
                });
                len = keys.length;

                probabilities_text += 'Classification Result: ' + res['highest'] + '</br>';
                probabilities_text += 'Probability/Confidence: ' + res['probability'] + '</br>';
                probabilities_text += 'Other Probabilities' + '</br>';
                for (var j = 1; j < len; j++) {
                    if (j >= 6) {
                        break;
                    }
                    name = keys[j];
                    probabilities_text += name + ': ' + res['probabilities'][name] + '</br>';


                }
                probabilities_text += '</p>';

                html_result += '<div class="col-md-6"><div id="result' + i + '">' + image_id_text + probabilities_text
                    + '</div></div>';

            }
            html_result += '</div></div>'
        }
        classification_div.prepend(html_result);
        if (classification_div.children().length > 30) {
            classification_div.children().last().remove();
        }
    });
});