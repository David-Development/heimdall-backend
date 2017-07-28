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
        var url = "/api/classifier/load/" + $(this).data('id');
        $.ajax({
            type: 'POST',
            url: url
        }).done(function () {
            location.reload();
        });
    });

    $('#model_table tbody').on('click', '.model-delete', function () {
        var url = '/api/classifier/delete/' + $(this).data('id');

        $.ajax({
            type: 'DELETE',
            url: url
        }).done(function (data) {
            console.log(data);
            location.reload();
        });

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


    $('#plotModal').on('show.bs.modal', function (event) {
        var button = $(event.relatedTarget); // Button that triggered the modal
        var model_id = button.data('id'); // Extract info from data-* attributes
        var url = '/api/model/' + model_id;
        var modal = $(this);
        var labels = [];
        $.getJSON(url, function (data) {

            $.each(data.labels, function(idx, lbl){
               labels.push(lbl.num + ": " + lbl.label);
            });

            modal.find('#confusion_plot').attr('src', data.confusion_url);
            modal.find('#learning_curve').attr('src', data.learning_curve_url);

            modal.find('#label_dict').text(labels);
        });
        // If necessary, you could initiate an AJAX request here (and then do the updating in a callback).
        // Update the modal's content. We'll use jQuery here, but you could use a data binding library or other methods instead.

        //modal.find('.modal-title').text('New message to ' + recipient)
        //modal.find('.modal-body input').val(recipient)
    });


    TaskUpdater.register_event(function (running) {
        $('#train_model').prop('disabled', running);

    });


})(jQuery);