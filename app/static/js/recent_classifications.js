/**
 * Created by Constantin on 28.07.2017.
 */


var remove_classification_row = function (div_id) {
    div = $('div[id=' + div_id + ']').remove();
};

var move_image = function (image_id, gallery_id) {
    var data = {};
    data.gallery_id = gallery_id;
    data.image_ids = [image_id];
    console.log(data);
    $.ajax({
        type: 'PUT',
        url: '/api/images/',
        data: JSON.stringify(data),
        contentType: 'application/json'
    }).done(function (data) {
        console.log(data);
        remove_classification_row(image_id)
    })
};

$('#classifications').on('click', '#correct_button', function () {
    var image_id = $(this).data('id');
    var gallery_id = $(this).attr('data-gallery-id');
    move_image(image_id, gallery_id);

    /*
     $.ajax({
     type: 'DELETE',
     url: url
     }).done(function (data) {
     console.log(data);
     location.reload();
     });
     */

});
$('#classifications').on('click', '#wrong_button', function () {
    var image_id = $(this).data('id');
    var gallery_id = $('#wrong_select_' + image_id).val();
    console.log(gallery_id);
    if (gallery_id !== null) {

        move_image(image_id, gallery_id);
    }
});
