/*

 */

#define RANGE 100

#include <iostream>
#include <string>
// binary trees
#include <vector>
#include <climits>
#include <unordered_map>
using namespace std;

void strhashing()
{
	string h1 = "Educba";
	cout << "string: " << h1 << endl;
	hash<string> hash_obj;
	cout << "hash: " << hash_obj(h1) << endl;
}

void customSort(int arr[], int n)
{
    int freq[RANGE];

    //memset(freq, 0, sizeof(freq));

    for (int i = 0; i < n; i++)
        freq[arr[i]]++;

    int k = 0;
    for (int i = 0; i < RANGE; i++)
        while (freq[i]--)
            arr[k++] = i;
}

void findFrequency(int A[], int n)
{
    int freq[n];

    for (int i = 0; i < n; i++)
        freq[i] = 0;

    for (int i = 0; i < n; i++)
        freq[A[i]]++;

    for (int i = 0; i < n; i++)
        if (freq[i])
            printf("%d appears %d times\n", i, freq[i]);
}

struct Node
{
    int data;
    Node* left, * right;

    Node(int data)
    {
        this->data = data;
        this->left = this->right = nullptr;
    }
};

void preorderTraversal(Node* root)
{
    if (root == nullptr)
        return;

    cout << root->data << " ";
    preorderTraversal(root->left);
    preorderTraversal(root->right);
}

void inorderTraversal(Node* root)
{
    if (root == nullptr)
        return;

    inorderTraversal(root->left);
    cout << root->data << " ";
    inorderTraversal(root->right);
}

void postorderTraversal(Node* root)
{
    if (root == nullptr)
        return;

    postorderTraversal(root->left);
    postorderTraversal(root->right);
    cout << root->data << ' ';
}

Node* buildTree(vector<int> const& inorder, int start, int end, unordered_map<int, int> map)
{
    if (start > end)
        return nullptr;

    int index = start;
    for (int j = start + 1; j <= end; j++)
        if (map[inorder[j]] < map[inorder[index]])
            index = j;

    Node* root = new Node(inorder[index]);
    root->left = buildTree(inorder, start, index - 1, map);
    root->right = buildTree(inorder, index + 1, end, map);

    return root;
}

Node* buildTree(vector<int> const& inorder, vector<int> const& level)
{
    int n = inorder.size();

    unordered_map<int, int> map;
    for (int i = 0; i < n; i++)
        map[level[i]] = i;

    return buildTree(inorder, 0, n-1, map);
}

int main()
{
	// test hash
    //strhashing();

    // OPTION 1 - sort array with duplicates
    int arr[] = {4, 2, 40, 10, 10, 1, 4, 2, 1, 10, 40};
    int n = sizeof(arr) / sizeof(arr[0]);

    cout << "pre-sort array: ";
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    cout << endl;

    customSort(arr, n);

    cout << "post-sort array: ";
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";
    cout << endl;

    // OPTION 2 - frequency
    /*int A[] = {2, 3, 3, 2, 1};
    int n = sizeof(A) / sizeof(A[0]);

    findFrequency(A, n);*/

    // OPTION 4 - inorder/postorder
    /*vector<int> inorder = {4,2,5,1,6,3,7};
    vector<int> postorder = { 4, 2, 7, 8, 5, 6, 3, 1 };

    Node* root = buildTree(inorder, postorder);
    cout << "inorder: ";
    inorderTraversal(root);
    cout << "postorder: ";
    postorderTraversal(root);

    // OPTION 5 - inorder/preorder
    vector<int> inorder = { 4,2,5,1,6,3,7 };
    vector<int> preorder = { 1, 2, 4, 3, 5, 7, 8, 6 };

    Node* root = buildTree(inorder, preorder);
    cout << "inorder: ";
    inorderTraversal(root);
    cout << "preorder: ";
    preorderTraversal(root);*/

	return 0;
}

