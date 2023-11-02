<?php
    // get data from form
    $noun1 = $_POST['noun1'];
    $verb1 = $_POST['verb1'];
    $adjective1 = $_POST['adjective1'];
    $noun2 = $_POST['noun2'];
    $verb2 = $_POST['verb2'];
    $adjective2 = $_POST['adjective2'];
    $noun3 = $_POST['noun3'];
    $noun4 = $_POST['noun4'];
    $noun5 = $_POST['noun5'];
    $verb3 = $_POST['verb3'];
    $adjective3 = $_POST['adjective3'];
?>



<!DOCTYPE html>
<html>

<head>
    <title>madlib product</title>

    <style>

        body {
            background-image: url("obamium.gif");
            background-size: 50%;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center center;
        }

        p {
            margin-left: 20px;
            display: block;
            width: 300px;
            padding: 1em;
        }

        strong {
            color: red;
        }

    </style>
</head>

<body>
    <main>
        <p>
        My fellow          <strong><span><?php echo $noun1; ?></span></strong>,
        we                 <strong><span><?php echo $verb1; ?></span></strong> 
        in a               <strong><span><?php echo $adjective1; ?></span></strong>
                           <strong><span><?php echo $noun2; ?></span></strong>, 
        but we must        <strong><span><?php echo $verb2; ?></span></strong>
        to make it even    <strong><span><?php echo $adjective2; ?></span></strong>. 
        Let us be guided 
        by our values of   <strong><span><?php echo $noun3; ?></span></strong>, 
                           <strong><span><?php echo $noun4; ?></span></strong>,
        and                <strong><span><?php echo $noun5; ?></span></strong>,
        and let us         <strong><span><?php echo $verb3; ?></span></strong>
        together towards a <strong><span><?php echo $adjective3; ?></span></strong>
        future for all. Yes we can, and yes we will.
        </p>
    </main>
</body>
</html>
