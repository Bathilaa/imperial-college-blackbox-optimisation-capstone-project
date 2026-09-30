# Datasheet

For the data behind the black-box optimisation capstone.

## Motivation

- The capstone has eight hidden functions.
- Each week I submit one query for each function and receive one result.
- The functions were created by Imperial College Executive Education for this course.
- The aim is to find the highest value for each function using a limited number of queries.

## Composition

- Each row contains one query and the result it returned.
- The course provided 175 readings at the start.
- After ten rounds I now have 255 readings.
- Each round adds eight readings, one for each function.
- The functions have between 2 and 8 inputs.
- Every input must be between 0 and 1.
- The data is stored in .npy files, one inputs file and one outputs file per function, under `data/function_1` to `data/function_8`.
- There are no missing results.

### Data coverage

- The search space is not evenly covered.
- For Function 1, which has two inputs, 34.5% of the space is within 0.1 of an existing reading.
- For Function 8, which has eight inputs, none of the tested space is within 0.1 of an existing reading, even with 49 readings.
- This shows how quickly coverage becomes difficult as the number of inputs increases.

## Collection process

- The course supplied the starting readings.
- I collected the rest through weekly submissions from August to September 2026.
- I submitted the queries through the course portal.
- Results arrived by email 1 to 3 days later.
- The queries were not selected randomly.
- Most were chosen because my Gaussian Process gave them the highest Expected Improvement.

## Preprocessing and uses

Changes made before modelling:

- Function 1: I model the size of each result rather than its signed value.
- Function 5: its results range from about 0.1 to more than 7,000, so I apply a log transformation when fitting the model.
- These changes are made only when fitting the models.
- The original results are kept unchanged in the dataset.

### How the data can be used

- It can be used to test different acquisition functions.
- It can be used to compare different modelling methods.
- It allows earlier rounds to be tested again without using new queries.

### How the data should not be used

- It should not be used to describe the full shape of the hidden functions.
- The data is concentrated around areas that my earlier models considered promising.
- Large parts of the search space have never been tested.

## Distribution and maintenance

- The data is available in my public GitHub repository.
- The course does not provide a licence for the hidden functions.
- I have therefore published only my own queries and the results I received.
- I have not published course material describing the functions.
- I will maintain the dataset for the length of the capstone.
- Each round is saved separately, so earlier versions can be recovered.
