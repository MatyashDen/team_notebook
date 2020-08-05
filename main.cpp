///////////////////////////////////////////////////////////////////////////
START VANYA START VANYA START VANYA START VANYA START VANYA START VANYA START VANYA START VANYA

@Vanya
//Segment Tree

class ST{
    int *a, *t, *o, n;
    inline void build(int v, int l, int r){
        o[v] = 0;
        if(l == r){
            t[v] = a[l];
        }
        else{
            int m = (r + l) >> 1;
            build(v + v + 1, l, m);
            build(v + v + 2, m + 1, r);
            t[v] = min(t[v + v + 1], t[v + v + 2]);
        }
    }
    inline void update(int v, int l, int r, int pos, int val){
        if(l == r){
            t[v] += val;
        }
        else{
            int m = (r + l) >> 1;
            if(pos <= m){
                update(v + v + 1, l, m, pos, val);
            }
            else{
                update(v + v + 2, m + 1, r, pos, val);
            }
            t[v] = min(t[v + v + 1], t[v + v + 2]);
        }
    }
    inline void push(int v){
        t[v] += o[v];
        if(v + v + 2 < 4 * n){
            o[v + v + 1] += o[v];
            o[v + v + 2] += o[v];
        }
        o[v] = 0;
    }
    inline void update(int v, int l, int r, int tl, int tr, int val){
        if(tl > tr) return;
        if(tl == l && tr == r){
            o[v] += val;
            push(v);
        }
        else{
            push(v);
            int m = (r + l) >> 1;
            update(v + v + 1, l, m, tl, min(m, tr), val);
            update(v + v + 2, m + 1, r, max(m + 1, tl), tr, val);
            t[v] = min(t[v + v + 1], t[v + v + 2]);
        }
    }
    inline int query(int v, int l, int r, int tl, int tr){
        push(v);
        if(tl > tr) return 1e18;
        if(l == tl && r == tr){
            return t[v];
        }
        int m = (r + l) >> 1;
        int t1 = query(v + v + 1, l, m, tl, min(tr, m));
        int t2 = query(v + v + 2, m + 1, r, max(tl, m + 1), tr);
        return min(t1, t2);
    }
public:
    ST(){}
    ST(int n) : n(n){
        t = new int[4 * n];
        o = new int[4 * n];
        a = new int[n];
        for(int i = 0; i < n; ++i) a[i] = 0;
        build(0, 0, n - 1);
    }
    ST(int n, int *a) : n(n), a(a){
        t = new int[4 * n];
        o = new int[4 * n];
        build(0, 0, n - 1);
    }
    void update(int pos, int val){
        update(0, 0, n - 1, pos, val);
    }
    void update(int l, int r, int val){
        update(0, 0, n - 1, l, r, val);
    }
    int query(int l, int r){
        return query(0, 0, n - 1, l, r);
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
@Vanya
//DSU

const int N = 2e5 + 7;
int p[N], sz[N];

void build_set(int n){
    for(int i = 0; i < n; ++i){
        p[i] = i;
        sz[i] = 1;
    }
}

int find_set(int v){
    if(v == p[v]) return v;
    return p[v] = find_set(p[v]);
}

bool union_sets(int x, int y){
    x = find_set(x);
    y = find_set(y);
    if(x != y){
        if(sz[x] > sz[y]){
            swap(x, y);
        }
        sz[y] += sz[x];
        sz[x] = 0;
        p[x] = y;
        return true;
    }
    else{
        return false;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
@Vanya
//modul

const int M = 998244353;

int add(int a, int b){
    a += b;
    if(a >= M) a -= M;
    if(a < 0) a += M;
    return a;
}

int mul(int a, int b){
    return a * (long long)b % M;
}

int bin(int a, int b){
    int res = 1;
    while(b > 0){
        if(b % 2 == 0){
            a = mul(a, a);
            b /= 2;
        }
        else{
            res = mul(res, a);
            --b;
        }
    }
    return res;
}

int inv(int a){
    return bin(a, M - 2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
@Vanya
//random

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
shuffle(a, a + n, rng);

///////////////////////////////////////////////////////////////////////////////////////////////////////
@Vanya
// multiply matrix

typedef vector < vector < int > > Matrix;

const int M = 1e9 + 7;

inline int add(int a, int b){
    a += b;
    if(a > M) a -= M;
    if(a < 0) a += M;
    return a;
}

inline int mul(int a, int b){
    return a * (long long)b % M;
}

inline Matrix mulMatrix(Matrix a, Matrix b){
    int n = b.size(), m =  min(a.size(), b[0].size());
    Matrix c(n, vector < int > (m));
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < m; ++j){
            for(int k = 0; k < n; ++k){
                c[i][j] = mul(c[i][j], mul(a[i][k], b[k][j]));
            }
        }
    }
    return c;
}

inline Matrix powMatrix(Matrix a, long long n){
    if(n == 1){
        return a;
    }
    if(n % 2 == 1){
        return mulMatrix(powMatrix(a, n - 1), a);
    }
    else{
        Matrix to = powMatrix(a, n / 2);
        return mulMatrix(to, to);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
@Vanya
//LCA

const int N = 1e5 + 7, P = 24;

int dist[N], p[N][P], tin[N], tout[N], T;
vector < int > g[N];

void dfs(int v, int pr = -1){
    tin[v] = ++T;
    p[v][0] = pr;
    for(int i = 1; i < P; ++i){
        if(p[v][i - 1] == -1){
            p[v][i] = -1;
        }
        else{
            p[v][i] = p[p[v][i - 1]][i - 1];
        }
    }
    for(int to : g[v]){
        if(to == pr) continue;
        dist[to] = dist[v] + 1;
        dfs(to, v);
    }
    tout[v] = ++T;
}

bool is(int x, int y){
    return tin[x] <= tin[y] && tout[x] >= tout[y];
}

int findLCA(int x, int y){
    if(is(x, y)) return x;
    if(is(y, x)) return y;
    for(int i = P - 1; i >= 0; --i){
        if(p[y][i] == -1) continue;
        if(!is(p[y][i], x)){
            y = p[y][i];
        }
    }
    return p[y][0];
}

END VANYA END VANYA END VANYA END VANYA END VANYA END VANYA END VANYA END VANYA END VANYA END VANYA
////////////////////////////////////////////////////////////////////////////////////////////////////////

Some useful tips from C++

rope<int> rp; (неявное ДД)

(faster than unordered_map<int, int>)
cc_hash_table<int, int> table;
gp_hash_table<int, int> table;

////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef tree<int, null_type, less<int>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
USAGE:

ordered_set X;
X.insert(8);
X.insert(16);

cout<<*X.find_by_order(4)<<endl; // 16
cout<<(end(X)==X.find_by_order(6))<<endl; // true

cout<<X.order_of_key(-5)<<endl;  // 0
cout<<X.order_of_key(4)<<endl;   // 2
cout<<X.order_of_key(400)<<endl; // 5

////////////////////////////////////////////////////////////////////////////////////////////////////////

//cerr << "Time elapsed: " << clock() / (double)CLOCKS_PER_SEC << endl;
typedef __uint128_t ui128;
cout<< __builtin_popcount (4);


////////////////////////////////////////////////////////////////////////////////////////////////////////////
TestingSystems :

////////////////////////////////////////////////////////////////////////////////////////////////
For windows:

stress.bat
@echo off

g++ -std=c++11 NewProgramist/main.cpp -o a.exe
g++ -std=c++11 brute/main.cpp -o brute.exe
g++ -std=c++11 gen/main.cpp -o gen.exe

set i=0

:loop
  gen.exe %i% > in
  a.exe < in > out
  brute.exe < in > out-brute

  fc out out-brute
  if errorlevel 1 goto fail

  echo OK
  set /a i=%i%+1
  goto loop

:fail
  echo Found failing test!
  PAUSE

////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////
For linux:

s.sh:

# compile all programs first
g++ brute.cpp -o brute
g++ a.cpp -o a
g++ gen.cpp -o gen

for((i = 1; ; ++i)); do
    echo $i
    {
        ./gen $i > int
        ./a < int > out1
    } || {
        break
    }
    ./brute < int > out2
    diff -w out1 out2 || break
    diff -w <(./a < int) <(./brute < int) || break
done

////////////////////////////////////////////////////////////////////////////////////////////////
For mac:
s.sh:

# compile all programs first
clang++ -std=c++11 -stdlib=libc++ ol/brute/brute.cpp -o brute
clang++ -std=c++11 -stdlib=libc++ ol/ol/main.cpp -o main
clang++ -std=c++11 -stdlib=libc++ ol/gen/gen.cpp -o gen

for((i = 1; ; ++i)); do
    echo $i
    {
        ./gen $i > int
        ./main < int > out1
    } || {
        break
    }
    ./brute < int > out2
    diff -w out1 out2 || break
    diff -w <(./main < int) <(./brute < int) || break
done
