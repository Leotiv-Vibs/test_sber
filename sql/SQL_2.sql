SELECT
SUBSTRING_INDEX(name,' ', 1) AS first_name,
SUBSTRING_INDEX(SUBSTRING_INDEX(name, ' ', 2),' ', -1) AS middle_name,
SUBSTRING_INDEX(name,' ', -1) AS last_name
FROM people;