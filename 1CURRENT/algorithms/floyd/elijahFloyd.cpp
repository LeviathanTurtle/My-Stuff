#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>

#define INFINITY 2000;

using namespace std;

int remove_v(string input);
void floyd(int length, const int W[6][6], int D[6][6], int P[6][6]);
void path(const int P[6][6], int from, int to);

int main()
{
	string file_name;
	ifstream input;
	int verts, edges;
	int W[6][6], D[6][6], P[6][6];
	int l_int, r_int;
	string l_str, r_str;
	int w;
	int num_paths;
	int in_deg, out_deg;
	cout << "Enter file name for first input." << endl;
	cin >> file_name;

	cout << endl;

	input.open(file_name);

	while (input.fail())
	{
		cout << "Your file name was invalid, try again." << endl;
		cin >> file_name;
		input.open(file_name);
	}

	input >> verts;
	input >> edges;

	for (int i = 0; i < 6; i++)
		for (int j = 0; j < 6; j++)
			W[i][j] = INFINITY;

	for (int i = 0; i < 11; i++)
	{
		input >> l_str;
		input >> r_str;
		input >> w;

		l_int = remove_v(l_str) - 1;
		r_int = remove_v(r_str) - 1;

		W[l_int][r_int] = w;
	}
	input >> num_paths;

	floyd(verts, W, D, P);

	cout << "W:" << endl;
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			cout << setw(4) << left << W[i][j] << " ";
		}
		cout << endl;
	}

	cout << endl;

	cout << "In Degrees and Out Degrees" << endl;

	for (int i = 0; i < 6; i++)
	{
		in_deg = 0;
		out_deg = 0;
		for (int j = 0; j < 6; j++)
		{
			if (W[i][j] != 2000)
			{
				in_deg++;
			}

			if (W[j][i] != 2000)
			{
				out_deg++;
			}
		}

		cout << "v" << i + 1 << " in degree = " << in_deg << " ";
		cout << "v" << i + 1 << " out degree = " << out_deg;
		cout << endl;
	}

	cout << endl;

	cout << "D: " << endl;
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			cout << setw(4) << left << D[i][j] << " ";
		}
		cout << endl;
	}

	cout << endl;

	cout << "P: " << endl;
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			cout << setw(4) << left << P[i][j] << " ";
		}
		cout << endl;
	}

	cout << endl;

	for (int i = 0; i < 3; i++)
	{
		input >> l_str;
		input >> r_str;

		l_int = remove_v(l_str) - 1;
		r_int = remove_v(r_str) - 1;

		cout << "Path from " << l_str << " to " << r_str << ": ";
		path(P, l_int, r_int);
		cout << endl;
	}

	input.close();
	return 0;
}

int remove_v(string input)
{
	int output;
	char holder = input[1];

	output = holder - '0';

	return output;
}

void floyd(int length, const int W[6][6], int D[6][6], int P[6][6])
{
	for (int i = 0; i < length; i++)
		for (int j = 0; j < length; j++)
			P[i][j] = 0;

	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			D[i][j] = W[i][j];
		}
	}

	for (int k = 0; k < length; k++)
	{
		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < length; j++)
			{
				if (D[i][j] > (D[i][k] + D[k][j]))
				{
					P[i][j] = k;
					D[i][j] = (D[i][k] + D[k][j]);
				}
			}
		}
	}
}

void path(const int P[6][6], int from, int to)
{
	if (P[from][to] != 0)
	{
		path(P, from, P[from][to]);
		cout << "v" << P[from][to] << " ";
		path(P, P[from][to], to);
	}
}