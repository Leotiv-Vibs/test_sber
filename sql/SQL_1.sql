with fibonacci(a, b) as
(
 select 1, 1
  union all
 select b, a+b from fibonacci where b < 1000000000
)
SELECT cast(a as varchar)+', ' AS [text()]
  FROM fibonacci
   FOR XML PATH ('')