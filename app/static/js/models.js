(function ($) {

    $('#model_table').DataTable({
        searching: false,
        pageLength: 30,
        lengthChange: false,
        "order": [[8, "desc"], [10, "desc"]]
    });


    $('#model_table tbody').on('click', '.model-change', function (e) {
        var data = $(this).data('id');
        console.log(e);
        console.log(data);
        var url = "/api/classifier/load/" + $(this).data('id');
        $.ajax({
            type: 'POST',
            url: url
        }).done(function () {
            location.reload();
        })
    });


})(jQuery);