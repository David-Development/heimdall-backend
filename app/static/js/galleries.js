(function ($) {

    var selected_images = [];

    var load_selected_gallery = function () {
        $('#control_div').removeClass('hidden');
        var gallery_id = $('#gallery_select').val();
        var url = '/api/gallery/' + gallery_id + '/images/';
        load_images(url);
        var text = $('#gallery_select option:selected').text();
        if (text === 'new' || text === 'unknown') {
            $('#delete_button').prop("disabled", true);
            $('#clear_button').show()
        }
        else {
            $('#delete_button').prop("disabled", false);
            $('#clear_button').hide()
        }
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

    $('#create_button').on('click', function () {
        var data = {};
        data.name = $('#new_gallery_name').val();

        $.ajax({
            type: 'POST',
            url: '/api/gallery/',
            data: JSON.stringify(data),
            contentType: 'application/json'
        }).done(function (data) {
            console.log(data);
            var option = $('<option />');
            option.val(data.id);
            option.text(data.name);
            console.log(option);
            $('#action_select').append(option);
            $('#gallery_select').append(option.clone());
            $('#gallery_select').attr('size', $('#gallery_select option').length);

        })
    });

    $('#delete_button').on('click', function () {
        var gallery_select = $('#gallery_select');
        var gallery_id = gallery_select.val();

        $.ajax({
            type: 'DELETE',
            url: '/api/gallery/' + gallery_id + '/'
        }).done(function (data) {
            $('#action_select option[value="' + gallery_id + '"]').remove();
            $('#gallery_select option:selected').remove();
            $('#gallery_select').attr('size', $('#gallery_select option').length);
            gallery_select.find('option:eq(0)').prop('selected', true);
            gallery_select.trigger('change');
        });
    });

    $('#clear_button').on('click', function () {
        var gallery_id = $('#gallery_select').val();
        console.log(gallery_id);

        $.ajax({
            type: 'POST',
            url: '/api/gallery/' + gallery_id + '/clear'
        }).done(function (data) {
            console.log(data);
            gallery_select.trigger('change');
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
            img.attr('title', image.date);
            img.addClass('galleryview');
            img.on('click', imageclick);
            image_div.append(img);
        });

    };
    var gallery_select = $('#gallery_select');
    gallery_select.on('change', load_selected_gallery);
    gallery_select.find('option:eq(0)').prop('selected', true);
    gallery_select.trigger('change');
    $('.galleryview').tooltip();


})(jQuery);