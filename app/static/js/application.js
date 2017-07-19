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
        var img_src = 'data:image/png;base64,' + result_obj['image'];
        var image_id = result_obj['image_id'];
        image_id_text = '<p>Image ID: ' + result_obj['image_id'] + '</p>';
        var img_text = '<div class="col-md-8"><img id=' + image_id + ' src=' + img_src + ' class="img-rounded"></div>';
        var html_result = '<div class="row margin_top liveresult">' + img_text;
        if (result_obj['classification']['predictions'].length === 0) {
            html_result += '<div class="col-md-4"><div id="result">' + image_id_text + ' ' +
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

                for (i = 0; i < len; i++) {
                    k = keys[i];
                    console.log(k + ':' + probabilities[k]);
                }
                probabilities_text += 'Classification Result: ' + res['highest'] + '</br>';
                probabilities_text += 'Probability/Confidence: ' + res['probability'] + '</br>';
                probabilities_text += 'Other Probabilities' + '</br>';
                for (i = 1; i < len; i++) {
                    if (i >= 6) {
                        break;
                    }
                    name = keys[i];
                    probabilities_text += name + ': ' + res['probabilities'][name] + '</br>';


                }
                probabilities_text += '</p>';

                html_result += '<div class="col-md-4"><div id="result">' + image_id_text + probabilities_text
                    + '</div></div></div>';

            }
        }
        classification_div.prepend(html_result);
    });
});