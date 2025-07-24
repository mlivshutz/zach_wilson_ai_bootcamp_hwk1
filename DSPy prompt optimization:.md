DSPy prompt optimization:
https://github.com/jpmajesty2025/prompt-optimization-dspy

https://github.com/LRCole3/DataExpert.io_Assignments
https://dspy.ai/tutorials/saving/

AdalFlow prompt optimization:
https://github.com/archithareddykommera/AIbootcamp_prompt_engineering

Hwk2:
2 RAG tests:
Does the endpoint return a response.
does the endpoint response contain any data


@lenin, thanks for tip. I searched and found some resource i'll read:
1. https://www.palantir.com/docs/foundry/ontology/core-concepts/
(Looks more detailed than neo4j docs)

2. https://community.openai.com/t/what-ontology-rag-and-graph-data-do-you-use-to-develop-intelligent-assistants/787860

3. LLM-assisted Ontology construction: https://arxiv.org/pdf/2404.03044

https://neo4j.com/blog/developer/knowledge-graph-vs-vector-rag/


Received notification from DBMS server: {severity: WARNING} {code: Neo.ClientNotification.Statement.UnknownPropertyKeyWarning} {category: UNRECOGNIZED} {title: The provided property key is not in the database} {description: One of the property names in your query is not available in the database, make sure you didn't misspell it or that the label is available when you run this statement in your application (the missing property name is: name)} {position: line: 3, column: 20, offset: 116} for query: 'cypher\nMATCH (nobelWinner)-[:WON]->(nobelPrize), (nobelWinner)-[:CONTRIBUTED_TO]->(contribution)\nRETURN nobelWinner.name, nobelPrize.name, contribution.description\n'

try 1
MATCH (p:Person)-[:WON]->(n:NobelPrize), (p)-[:CONTRIBUTED_TO]->(c:Contribution)
RETURN p.name AS NobelPrizeWinner, c.description AS ContributionToScience

try 2
MATCH (p:Person)-[:WON]->(n:NobelPrize)-[:FOR]->(c:Contribution)
RETURN p.name AS Winner, c.description AS Contribution