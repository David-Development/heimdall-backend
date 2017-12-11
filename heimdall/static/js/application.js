var TaskUpdater = function () {

    var task_url = '/api/tasks/';
    var timeout;
    var running = false;
    var change_callbacks = [];
    var timeout_obj;

    var call_change_callbacks = function () {
        $.each(change_callbacks, function (index, callback) {
            callback(running);
        });
    };

    var handledata = function (data) {
        var icon_elem = $('#state_icon');
        var state_elem = $('#state');
        var step_elem = $('#step');
        var current_elem = $('#current');
        var base_classes = 'fa fa-2x fa-fw';

        if ($.isEmptyObject(data) || data['app.tasks.train_recognizer'] === typeof 'undefined') {
            running = false;
            call_change_callbacks();
            timeout = 10000;
            icon_elem.removeClass('fa-spin fa-cog fa-check');
            icon_elem.addClass(base_classes);
            icon_elem.addClass('fa-exclamation');

            state_elem.text('No Model trained since last start...');
            step_elem.text('');
            current_elem.text('');
        } else {
            var task_info = data['app.tasks.train_recognizer'];

            var task = task_info.details;
            if (task.state === 'STARTED') {
                running = true;
                call_change_callbacks();
                timeout = 1000;
                icon_elem.removeClass('fa-check fa-exclamation');
                icon_elem.addClass(base_classes);
                if (!icon_elem.hasClass('fa-spin')) {
                    icon_elem.addClass('fa-cog fa-spin');
                }

                state_elem.text('Started');
                step_elem.text(task.step);

                var current_elem_text = '';

                if (task.step === 'Augmenting') {
                    current_elem_text += "Person ";
                } else {
                    current_elem_text += "Image ";
                }

                current_elem_text += task.current + " of " + task.total;
                current_elem.text(current_elem_text);
            }
            else {
                running = false;
                call_change_callbacks();
                timeout = 10000;
                icon_elem.removeClass('fa-spin fa-cog fa-exclamation');
                icon_elem.addClass(base_classes);
                icon_elem.addClass('fa-check');

                state_elem.text('Training of model finished');
                step_elem.text('');
                current_elem.text('');
            }
        }

        console.log("Next update in: ", timeout);


    };

    var callback = function (data) {
        handledata(data);
        timeout_obj = setTimeout(get_tasks, timeout);

    };

    var get_tasks = function () {
        clearTimeout(timeout_obj);
        $.getJSON(task_url, callback);
    };


    return {
        update: function () {
            get_tasks();
        },
        register_event: function (callback) {
            change_callbacks.push(callback);
        }
    }
}();