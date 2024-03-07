#include<iostream>
#include<random>
#include <fstream>
#include <iomanip>
#include <string>
#include<chrono>
#include <Eigen/Dense>


const int MATRIX_SIZE = 500;

double** getMatrix(int n, int m, bool isEmpty)
{
	double** matrix = new double* [n];
	if (!isEmpty)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dist(100.0, 1000.0);
		for (int i = 0; i < n; i++)
		{
			matrix[i] = new double[m];
			for (int j = 0; j < m; j++)
			{
				matrix[i][j] = dist(gen);
			}
		}
	}
	else
	{
		for (int i = 0; i < n; i++)
		{
			matrix[i] = new double[m];
		}
	}
	return matrix;
}

void mulMatrices(double** matrix1, double** matrix2, double** resultMatrix, int& operationsCounter)
{
	for (int i = 0; i < MATRIX_SIZE; ++i)
	{
		for (int j = 0; j < MATRIX_SIZE; ++j)
		{
			resultMatrix[i][j] = 0.0;
		}
	}

	for (int i = 0; i < MATRIX_SIZE; ++i)
	{
		for (int j = 0; j < MATRIX_SIZE; ++j)
		{
			for (int k = 0; k < MATRIX_SIZE; ++k)
			{
				resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
				operationsCounter++;
			}
		}
	}
}

Eigen::MatrixXd convertToEigen(double** matrix, int n, int m) 
{
	Eigen::MatrixXd mat(n, m);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			mat(i, j) = matrix[i][j];
		}
	}
	return mat;
}

bool compareMatrices(double** matrix1, double** matrix2, Eigen::MatrixXd& eigenResult, int n, int m)
{
	Eigen::MatrixXd result = convertToEigen(matrix1, n, m) * convertToEigen(matrix2, n, m);
	return result.isApprox(eigenResult, 1e-10);
}

void writeJsonToFile(const std::string& filePath, int operationsCounter, bool isCorrect, double durationSeconds) 
{
	std::string json = "{\n";
	json += "  \"OperationsCount\": " + std::to_string(operationsCounter) + ",\n";
	json += "  \"IsMultiplicationCorrect\": " + std::string(isCorrect ? "true" : "false") + ",\n";
	json += "  \"DurationSeconds\": " + std::to_string(durationSeconds) + "\n";
	json += "}\n";

	std::ofstream outFile(filePath);
	if (outFile.is_open()) {
		outFile << json;
		outFile.close();
	}
	else {
		std::cerr << "Unable to open file for writing JSON data.\n";
	}
}

int main()
{

	double** matrix1 = getMatrix(MATRIX_SIZE, MATRIX_SIZE, false);
	double** matrix2 = getMatrix(MATRIX_SIZE, MATRIX_SIZE, false);
	double** resultMatrix = getMatrix(MATRIX_SIZE, MATRIX_SIZE, true);

	int operatoinsCounter = 0;

	auto start = std::chrono::high_resolution_clock::now();
	mulMatrices(matrix1, matrix2, resultMatrix, operatoinsCounter);
	auto stop = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	Eigen::MatrixXd eigenResult = convertToEigen(resultMatrix, MATRIX_SIZE, MATRIX_SIZE);

	bool isCorrect = compareMatrices(matrix1, matrix2, eigenResult, MATRIX_SIZE, MATRIX_SIZE);

	std::string filePath = "..//matrixData.json"; 
	writeJsonToFile(filePath, operatoinsCounter, isCorrect, duration.count());

	return 0;
}