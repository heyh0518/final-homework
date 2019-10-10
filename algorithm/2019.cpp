#include<iostream>
#include <string>
#include<cstdio>
#include<cstring>
#include<cmath>
#include<algorithm>
using namespace std;

/*
3-1 begin
*/
using namespace std;
int n;
int time_a = 0;
int min_t = 0x3f3f3f;
int a[1024];
int b[1024];
int F[1024][1024];

int min(int a,int b){
	if (a>b)
		return b;
	else 
		return a;
}
int max(int a,int b){
	if (a<b)
		return b;
	else 
		return a;
}

void inPut()
{
    scanf("%d", &n);
    for(int i = 1; i <= n; ++i)
    {
     scanf("%d", &a[i]);
     time_a += a[i];
    }
    for(i = 1; i <= n; ++i)
        scanf("%d", &b[i]);
}
void solve()
{
    memset(F, 0, sizeof(F));
    for(int i = 1; i <= n; ++i)
    {
        for(int j = 0; j <= time_a; ++j)
        {
            if(j > a[i])
            {
                F[i][j] = min(F[i - 1][j - a[i]], F[i - 1][j] + b[i]);
            }
            else
            {
                F[i][j] = F[i - 1][j] + b[i];
            }
        }
    }
    int temp;
    for(i = 0; i <= time_a; ++i)
    {
        temp = max(F[n][i], i);
        if(min_t > temp)
            min_t = temp;
    }
}
/*
3-1 end
----------------------------------------------------------------------------------
5-4 begin
*/
#define N 100
int P[N][N],Q[N][N];
int x[N];
//int n;
int opt[N];
int tempValue=0,maxValue=0;
 
void compute(){
	tempValue = 0;
	for(int i=1;i<=n;i++){
		tempValue += P[i][x[i]]*Q[x[i]][i];
	}
	if(tempValue>maxValue){
		maxValue = tempValue;
		for(int i=1;i<=n;i++){
			opt[i] = x[i];
		}
	}
}
 
void traceback(int t){
	int i,temp;
	if(t>n){
		compute();
	}
	for(i=t;i<=n;i++){
		temp = x[i];
		x[i] = x[t];
		x[t] = temp;
		traceback(t+1);
		temp = x[i];
		x[i] = x[t];
		x[t] = temp;		
	}
}


/*
5-4 end
----------------------------------------------------------------------------------
4-11 begin
*/
int deleteK()
{
	int s;
	std::string n;
    bool flag=0;
	std::cin>>n;
    int lenn=n.length();
    scanf("%d",&s);
    for(int j=1;j<=s;j++)
		for(int i=0;i<n.length()-1;i++){
			if(n[i+1]<n[i]||n[i]==n.length()-1){
				n.erase(i,1);
				break;
			}
		}
    while(n[0]=='0'&&n[1]){
        n.erase(0,1);
    }
    for(int i=0;i<=n.length()-1;i++)
        printf("%c",n[i]);
    printf("\n");
	return 1;
}
/*
4-11 end
----------------------------------------------------------------------------------
2-8 begin
*/
long S(long m,long n)
{
    if(m==1)
        return 1;
    if(m==n)
        return 1;
    else
        return S(m-1,n-1)+S(m,n-1)*m;
}

void getParam(){
	long n,i;
    while(cin>>n>>i)
    {
        long sum=0;
		sum+=S(i,n);
        cout<<sum<<endl;
    }
}
/*
2-8 end
*/
