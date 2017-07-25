(function ($) {

    $('#model_table').DataTable({
        searching: false,
        pageLength: 100,
        lengthChange: false,
        "order": [[8, "desc"], [0, "desc"]],
        "columnDefs": [
            {"orderable": false, "targets": -1}
        ]
    });


    $('#model_table tbody').on('click', '.model-change', function () {
        var data = $(this).data('id');
        var url = "/api/classifier/load/" + $(this).data('id');
        $.ajax({
            type: 'POST',
            url: url
        }).done(function () {
            location.reload();
        })
    });

    $('#train_model').on('click', function () {
        $('#train_model').prop('disabled', true);
        $.getJSON('/api/recognizer/train/', function (data) {
            console.log(data);

        }).done(function () {
            //Give Task one second to get initialized
            setTimeout(TaskUpdater.update, 1000);
        });
    });


    TaskUpdater.register_event(function (running) {
        $('#train_model').prop('disabled', running);

    });


})(jQuery);