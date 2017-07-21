(function ($) {

    var selected_images = [];

    var load_selected_gallery = function () {
        $('#control_div').removeClass('hidden');
        var gallery_id = $('#gallery_select').val();
        var url = '/api/gallery/' + gallery_id + '/images/';
        load_images(url);
    };


    var imageclick = function () {
        var img = $(this);
        if (img.hasClass('selected')) {
            var removeItem = img.attr('id');
            selected_images = $.grep(selected_images, function (value) {
                return value !== removeItem;
            });
        }
        else {
            selected_images.push(img.attr('id'));
        }
        img.toggleClass('selected');
    };

    $('#move_button').on('click', function () {
        var data = {};
        data.gallery_id = $('#action_select').val();
        data.image_ids = selected_images;

        $.ajax({
            type: 'PUT',
            url: '/api/images/',
            data: JSON.stringify(data),
            contentType: 'application/json'
        }).done(function (data) {
            console.log(data);
            load_selected_gallery();
        }).success(function () {
            selected_images = [];
        });
    });

    var load_images = function (url) {
        var images = [];
        $.getJSON(url, {}, function (data) {
            $.each(data, function (index, img) {
                images.push(img);
            });
            set_images(images);
        });
    };

    var set_images = function (images) {
        var image_div = $('#images');
        selected_images = [];
        image_div.find('img').remove();
        $.each(images, function (index, image) {
            var img = $('<img />');
            img.attr('src', image.url);
            img.attr('id', image.id);
            img.addClass('galleryview');
            img.on('click', imageclick);
            image_div.append(img);
        });

    };

    $('#gallery_select').on('change', load_selected_gallery);


})(jQuery);