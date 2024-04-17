#include<iostream>
#include<random>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include<chrono>
#include<numeric>
#include <Eigen/Dense>
#include <omp.h>



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

void mulMatrices(double** matrix1, double** matrix2, double** resultMatrix, long long& operationsCounter, int matrixSize)
{
	for (int i = 0; i < matrixSize; ++i)
	{
		for (int j = 0; j < matrixSize; ++j)
		{
			resultMatrix[i][j] = 0.0;
		}
	}

	#pragma omp parallel for collapse(2)
	for (int i = 0; i < matrixSize; ++i)
	{
		for (int j = 0; j < matrixSize; ++j)
		{
			for (int k = 0; k < matrixSize; ++k)
			{
				resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
			}
			#pragma omp atomic
			operationsCounter++;
		}
	}

	#pragma omp parallel
	{
		#pragma omp master
		{
			std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
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



int main()
{
	const int repetitions = 10;
	std::vector<double> averageDurations;
	std::vector<long long> averageOperations;
	const std::vector<int> threadCounts = { 1, 2, 4, 10, 20 };

	std::ofstream outFile("./performance_openMP.csv");
	outFile << "MatrixSize, ThreadCount, MeanDuration, ConfidenceIntervalLower, ConfidenceIntervalUpper\n";

	for (int matrixSize = 50; matrixSize <= 800; matrixSize += 50)
	{
		long long totalDuration = 0;

		for (int threads : threadCounts)
		{
			std::vector<double> durations;
			for (int i = 0; i < repetitions; ++i)
			{
				omp_set_num_threads(threads);

				double** matrix1 = getMatrix(matrixSize, matrixSize, false);
				double** matrix2 = getMatrix(matrixSize, matrixSize, false);
				double** resultMatrix = getMatrix(matrixSize, matrixSize, true);

				long long operationsCounter = 0;

				auto start = std::chrono::high_resolution_clock::now();
				mulMatrices(matrix1, matrix2, resultMatrix, operationsCounter, matrixSize);
				auto stop = std::chrono::high_resolution_clock::now();

				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

				durations.push_back(duration);


				Eigen::MatrixXd eigenResult = convertToEigen(resultMatrix, matrixSize, matrixSize);
				if (i == 0)
				{
					bool isCorrect = compareMatrices(matrix1, matrix2, eigenResult, matrixSize, matrixSize);
					if (!isCorrect)
					{
						std::cerr << "Matrix multiplication is incorrect!" << std::endl;
					}
				}

				for (int j = 0; j < matrixSize; j++)
				{
					delete[] matrix1[j];
					delete[] matrix2[j];
					delete[] resultMatrix[j];
				}
				delete[] matrix1;
				delete[] matrix2;
				delete[] resultMatrix;
			}

		double mean = std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
		double sq_sum = std::inner_product(durations.begin(), durations.end(), durations.begin(), 0.0);
		double stdev = std::sqrt(sq_sum / durations.size() - mean * mean);
		double errorMargin = 1.96 * stdev / std::sqrt(durations.size());

		outFile << matrixSize << ", " << threads << ", " << mean << ", " << (mean - errorMargin) << ", " << (mean + errorMargin) << "\n";

		std::cout << "multiplicated " << matrixSize << " threads " << threads << " mean " << mean << std::endl;
		}
	}

	outFile.close();

	return 0;
}