-- Create the my_guitar_shop1 database
DROP DATABASE IF EXISTS whereabouts;
CREATE DATABASE whereabouts;
USE whereabouts;  -- MySQL command

-- create the table - CATEGORIES
CREATE TABLE actions (
  actionID       INT(11)        NOT NULL   AUTO_INCREMENT,
  actionName     VARCHAR(255)   NOT NULL,
  PRIMARY KEY (actionID)
);

-- create the table - PRODUCTS
CREATE TABLE places (
  placeID        INT(11)        NOT NULL   AUTO_INCREMENT,
  productName      VARCHAR(255)   NOT NULL,
  PRIMARY KEY (placeID)
);

-- create the table - 
CREATE TABLE people (
  peopleID        INT(11)        NOT NULL   AUTO_INCREMENT,
  peopleFName      VARCHAR(255)   NOT NULL,
  peopleLName      VARCHAR(255)   NOT NULL,
  peopleEmail      VARCHAR(255)   NOT NULL,
  PRIMARY KEY (peopleID)
);

-- insert data into the database
INSERT INTO actions VALUES
(1, 'running from class'),
(2, 'sleeping in'),
(3, 'partying hard'),
(4, 'getting arrested'),
(5, 'eating'),
(6, 'procrastinating his work'),
(7, 'buying half of publix'),
(8, 'gambling his life savings'),
(9, 'busy gaming'),
(10, 'robbing a bank');

INSERT INTO places VALUES
(1, 'Guatemala'),
(2, 'Space'),
(3, 'your walls'),
(4, "your mother's butt"),
(5, 'The Mariana Trench'),
(6, 'Seattle'),
(7, 'Las Vegas'),
(8, 'underneath your shoe'),
(9, 'an exclusive, high-end basement'),
(10, 'Kilpisjarvi');

-- create the users and grant priveleges to those users
GRANT SELECT, INSERT, DELETE, UPDATE
ON whereabouts.*
TO mgs_user@localhost
IDENTIFIED BY 'pa55word';

GRANT SELECT
ON places
TO mgs_tester@localhost
IDENTIFIED BY 'pa55word';
