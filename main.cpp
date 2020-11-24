// teamNotebook
// Defines, includes and namespaces(begin cpp file):
 
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
#include <ext/rope>
 
#define ll long long
#define ll128 __uint128_t
#define ld long double
#define pll pair <ll, ll>
 
#define vll vector <ll>
#define vld vector<ld>
#define vpll vector<pll>
 
#define vvll vector <vll>
 
#define rep(i, a, b) for(ll i = (ll)a; i < (ll)b; i++)
#define per(i, a, b) for(ll i = (ll)a - 1; i >= (ll)b; --i)
 
#define endl "\n"
#define pb push_back
#define pf push_front
 
#define all(v) (v).begin(), (v).end()
#define rall(v) (v).rbegin(), (v).rend()
 
#define sorta(v) sort(all(v))
#define sortd(v) sort(rall(v))
 
#define debug if (1)
#define log(val) debug {cout << "\n" << #val << ": " << val << "\n";}
 
#define ios ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);
#define file(name) freopen(name".in", "r", stdin); freopen(name".out", "w", stdout);
#define FILE freopen("input.txt", "r", stdin); freopen("output.txt", "w", stdout);
 
#define mod (ll)(1e9 + 7)
#define inf (mod * mod)
 
using namespace std;
using namespace __gnu_cxx;
using namespace __gnu_pbds;
 
ostream & operator << (ostream & out, vll & a) {
    for(auto i : a) out << i << " ";
    return out;
}
 
istream & operator >> (istream & in, vll & a) {
    for(auto &i : a) in >> i;
    return in;
}

const ll N = 2e5 + 5;
 
Algorithms, data structure
 
For adding algorithm or data structure use this form please:
 
////////////////////////////////////////////////
@yourNickname
Some text about pasted code
May be link of the problem where you use this code
 
Code
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
SegmentTree
 
struct Node {
    ll val;
 
    Node() : val(0LL) {}
    Node(ll value) : val(value) {}
};
 
const ll SZ = 2e5;
ll a[SZ];
Node t[4 * SZ];
ll add[4 * SZ];
 
struct SegTree {
 
    ll n;
    ll lazyOp;
    // 1 - if sum lazyOp
    // 2 - if max, min lazyOp
 
    ll passiveEl = 0LL; // getOp(a, passiveEl) = a
    ll neitralEl = 0LL; // if(b != neitralEl) a = b;(for lazy)
 
    SegTree(ll sz, ll lazy) {
        n = sz; lazyOp = lazy;
    }
 
    void input() {
        rep(i, 0, n) {
            cin >> a[i];
        }
    }
 
    Node getOp(Node left, Node right) {
        return Node(max(left.val, right.val));
    }
 
    void pull(ll v) {
        t[v] = getOp(t[v * 2 + 1], t[v * 2 + 2]);
    }
 
    void build(ll v, ll l, ll r) {
        if(l == r) {
            t[v] = Node(a[l]);
            return;
        }
 
        ll m = (l + r) / 2;
        build(v * 2 + 1, l, m);
        build(v * 2 + 2, m + 1, r);
 
        pull(v);
    }
 
    //////////////////////////////////////
    //Lazy operation
    ll pullChildOp(ll val, ll segSize) {
        if(lazyOp == 1) return val * segSize;
        else if(lazyOp == 2) return val;
        assert(false);
    }
 
    void push(ll v, ll l, ll m, ll r) {
        // Change += on =
        if(add[v] != neitralEl) {
            add[v * 2 + 1] += add[v];
            add[v * 2 + 2] += add[v];
 
            t[v * 2 + 1].val += pullChildOp(add[v], m - l + 1);
            t[v * 2 + 2].val += pullChildOp(add[v], r - m);
 
            add[v] = neitralEl;
        }
    }
    //////////////////////////////////////
 
    void updateSeg(ll v, ll tl, ll tr, ll l, ll r, ll val) {
        if(l > r) return;
        if(l == tl && r == tr) {
            // Change += on =
            t[v].val += pullChildOp(val, r - l + 1);
            add[v] += val;
            return;
        }
 
        ll tm = (tl + tr) / 2;
 
        push(v, tl, tm, tr);
 
        updateSeg(v * 2 + 1, tl, tm, l, min(tm, r), val);
        updateSeg(v * 2 + 2, tm + 1, tr, max(tm + 1, l), r, val);
 
        pull(v);
    }
 
    void updateEl(ll v, ll tl, ll tr, ll pos, Node val) {
        if(tl == tr) {
            t[v] = val;
            return;
        }
 
        ll tm = (tl + tr) / 2;
 
        if(pos <= tm) {
            updateEl(v * 2 + 1, tl, tm, pos, val);
        } else {
            updateEl(v * 2 + 2, tm + 1, tr, pos, val);
        }
 
        pull(v);
    }
 
    Node get(ll v, ll tl, ll tr, ll l, ll r) {
 
        if(l > r) return passiveEl;
        if(tl == l && r == tr) return t[v];
 
        ll tm = (tl + tr) / 2;
 
        if(lazyOp) push(v, tl, tm, tr);
        return getOp(get(v * 2 + 1, tl, tm, l, min(tm, r)),
                        get(v * 2 + 2, tm + 1, tr, max(tm + 1, l), r));
    }
 
    //Simply operations
    void updateSeg(ll l, ll r, ll val) {
        assert(lazyOp);
        updateSeg(0, 0, n - 1, l, r, val);
    }
 
    void updateEl(ll pos, ll val) {
        updateEl(0, 0, n - 1, pos, val);
    }
 
    Node get(ll l, ll r) {
        return get(0, 0, n - 1, l, r);
    }
 
    void build() {
        build(0, 0, n - 1);
    }
 
};
 
////////////////////////////////////////////////
 
 
 
 
////////////////////////////////////////////////
@Bogdan
Cartesian Tree in array
 
struct Node {
 
    ll priority, cnt, val, min;
    bool rev;
    Node * l, * r, *p;
    Node(ll n) : priority(rand()), cnt(1), val(n), min(n), rev(0), l(nullptr), r(nullptr), p(nullptr) {};
 
};
 
void buildTree(Node *& t, string & s) {
 
    Node * last = new Node((ll)(s[0] - '0'));;
    rep(i, 1, s.size()) {
 
        auto cur = new Node((ll)(s[i] - '0'));
        if(cur -> priority < last -> priority) {
            last -> r = cur;
            cur -> p = last;
            last = cur;
        } else {
            while(last -> p && last -> priority < cur -> priority) {
                last = last -> p;
            }
            if(last -> priority < cur -> priority) {
                cur -> l = last;
                last -> p = cur;
                last = cur;
            } else {
                cur -> l = last -> r;
                cur -> p = last;
                last -> r = cur;
                last = cur;
            }
        }
    }
 
    while(last -> p) {
        last = last -> p;
    }
    t = last;
}
 
ll cnt(Node * t) {
    if(!t) return 0;
    return t -> cnt;
}
 
ll minNode(Node * t) {
    if(!t) return 2e9;
    return t -> min;
}
 
void update(Node * t) {
    if(!t) return;
    t -> cnt = 1 + cnt(t -> l) + cnt(t -> r);
    t -> min = min({t -> val, minNode(t -> l), minNode(t -> r)});
}
 
void push(Node * t) {
    if(t && t -> rev) {
        t -> rev = 0;
        if(t -> l) t -> l -> rev ^= 1;
        if(t -> r) t -> r -> rev ^= 1;
        swap(t -> l, t -> r);
    }
}
 
void print(Node * t) {
    if(!t) return;
    print(t -> l);
    cout << t -> val << " " << t -> min << " " << t -> cnt << "   ";
    print(t -> r);
}
 
void split(Node * t, Node *& l, Node *& r, ll pos) {
    push(t);
    if(!t) {
        l = r = nullptr;
        update(l);
        update(r);
    }
    else {
        if(pos <= cnt(t -> l)) {
            split(t -> l, l, t -> l, pos);
            r = t;
            update(r);
        } else {
            split(t -> r, t -> r, r, pos - cnt(t -> l) - 1);
            l = t;
            update(l);
        }
    }
}
 
void merge(Node * l, Node * r, Node *& t) {
    push(l);
    push(r);
    if(!l || !r) {
        t = l ? l : r;
    } else {
        if(l -> priority > r -> priority) {
            merge(l -> r, r, l -> r);
            t = l;
        } else {
            merge(l, r -> l, r -> l);
            t = r;
        }
    }
    update(t);
}
 
void insert(Node *& t, Node * cur, ll pos) {
    Node * l, * r;
    split(t, l, r, pos);
    merge(l, cur, t);
    merge(t, r, t);
}
 
ll getMin(Node * t, ll l, ll r) {
    Node * l1, *r1, *r2;
    split(t, l1, r1, r);
    split(l1, l1, r2, l - 1);
    ll ans = minNode(r2);
    merge(l1, r2, t);
    merge(t, r1, t);
    return ans;
}
 
void rev(Node *& t, ll l, ll r) {
    Node * l1, * r1, * r2;
    split(t, l1, r1, r);
    split(l1, l1, r2, l - 1);
    if(r2) r2 -> rev ^= 1;
    merge(l1, r2, t);
    merge(t, r1, t);
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
Cartesian Tree(ordinary)
 
void insert (pitem &t, pitem it)
{
  if (!t)
    t = it;
  else if (it->Priority > t->Priority)
    split (t, it->Key, it->l, it->r),  t = it;
  else
    insert (it->Key < t->Key ? t->l : t->r, it);
}
 
void erase (pitem &t, int Key)
{
  if (t->Key == Key)
    merge (t->l, t->r, t);
  else
    erase (Key < t->Key ? t->l : t->r, Key);
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bohdan
// Paralell binary search
// Main function from problem: https://atcoder.jp/contests/agc002/tasks/agc002_d
// Full solution: https://pastebin.com/rsYrychi

int main() {
    ll n, m;
    cin >> n >> m;
    vector<pll> edges;
    rep(i, 0, m) {
        ll a, b;
        cin >> a >> b;
        edges.pb(make_pair(a - 1, b - 1));
    }
    ll q;
    cin >> q;
    vector<Query> qs(q);
    rep(i, 0, q) {
        cin >> qs[i].x >> qs[i].y >> qs[i].z;
        qs[i].x--;
        qs[i].y--;
    }

    per(bit, 20, 0) {
        rep(i, 0, n) {
            add(i);
        }
        rep(i, 0, m) {
            pos[i].clear();
        }
        rep(i, 0, q) {
            ans[i] += (1 << bit);
            if (ans[i] < m) {
                pos[ans[i]].pb(i);
            } else {
                ans[i] -= (1 << bit);
            }
        }

        rep(i, 0, m) {
            for (auto id : pos[i]) {
                auto query = qs[id];
                ll sum = 0;
                if (get(query.x) == get(query.y)) {
                    sum = s[get(query.x)];
                } else {
                    sum = (s[get(query.x)] + s[get(query.y)]);
                }
                if (sum >= query.z) {
                    ans[id] -= (1 << bit);
                }
            }
            unite(edges[i].first, edges[i].second);
        }
    }

    rep(i, 0, q) {
        cout << ans[i] + 1 << endl;
    }
    return 0;
}

////////////////////////////////////////////////


////////////////////////////////////////////////
@Bogdan
Persistent Treap
e-olymp.com/ru/submissions/6418681
 
const ll SZ = 10000000;
 
ll val[SZ], L[SZ], R[SZ], cnt[SZ];
 
ll gl = 0;
 
ll newNode(ll from) {
    gl++;
    val[gl] = val[from];
    L[gl] = L[from];
    R[gl] = R[from];
    cnt[gl] = cnt[from];
    return gl;
}
 
ll newNode(ll from, ll cur) {
    gl++;
    val[gl] = from;
    L[gl] = 0;
    R[gl] = 0;
    cnt[gl] = 1;
    return gl;
}
 
ll N;
ll K;
const ll md = 1e9 + 7;
 
unsigned int seed = 162146;
 
ll cntt(ll index) {
    if (!index) {
        return 0;
    }
    return cnt[index];
}
 
void update(ll index) {
    if (index) {
        cnt[index] = cntt(L[index]) + cntt(R[index]) + 1;
    }
}
 
ll merge_t(ll l, ll r) {
    ll ptrn = 0;
    if (!l) {
        ptrn = newNode(r);
        return ptrn;
    }
    if (!r) {
        ptrn = newNode(l);
        return ptrn;
    }
    if ((rand_r(&seed) % (cnt[l] + cnt[r])) > cnt[r]) {
        ptrn = newNode(l);
        R[ptrn] = merge_t(R[ptrn], r);
        update(ptrn);
        return ptrn;
    } else {
        ptrn = newNode(r);
        L[ptrn] = merge_t(l, L[ptrn]);
        update(ptrn);
        return ptrn;
    }
}
 
std::pair<ll, ll> split_t(ll t, int x) {
    if (t == 0) {
        return std::make_pair(0, 0);
    }
    ll cur = newNode(t);
    int indexx = cntt(L[t]) + 1;
    ll l, r;
    if (indexx <= x) {
        auto cur5 = split_t(R[cur], x - indexx);
        R[cur] = cur5.first;
        l = cur;
        r = cur5.second;
        update(l);
    } else {
        auto cur5 = split_t(L[cur], x);
        L[cur] = cur5.second;
        r = cur;
        l = cur5.first;
        update(r);
    }
    return std::make_pair(l, r);
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
Persistent Segment Tree + Persistent Array
Moscow Pre-finals Workshops(Day 5):
official.contest.yandex.com/mw2020prefinals/contest/18056/problems/
 
full code for problem PersistentQueue:
pastebin.com/JMxAT200
 
const ll SZ = 4000000;
ll N = 200010;
ll L[SZ], R[SZ], val[SZ];
 
struct PersistentArray {
 
    ll gl = 1;
    vll a;
 
    PersistentArray() {}
 
    PersistentArray(vll & newa) {
        a = newa;
    }
 
    ll cop(ll v) {
        if (!v) return v;
        val[gl] = val[v];
        L[gl] = L[v];
        R[gl] = R[v];
        return gl++;
    }
 
    ll build(ll tl = 0, ll tr = N) {
        if(tl == tr) {
            val[gl] = 0;
            L[gl] = R[gl] = 0;
            return gl++;
        }
        ll tm = (tl + tr) >> 1;
        ll v = gl++;
        L[v] = build(tl, tm);
        R[v] = build(tm + 1, tr);
        return v;
    }
 
    ll update(ll v, ll tl, ll tr, ll pos, ll el) {
        if (tl == tr) {
            val[v] = el;
            return v;
        }
        ll tm = (tl + tr) / 2;
        if (pos <= tm) {
            L[v] = cop(L[v]);
            return update(L[v], tl, tm, pos, el);
        } else {
            R[v] = cop(R[v]);
            return update(R[v], tm + 1, tr, pos, el);
        }
    }
 
 
    ll query(ll v, ll tl, ll tr, ll pos) {
        while (tl < tr) {
            int tm = (tl + tr) >> 1;
            if(pos <= tm) {
                v = L[v];
                tr = tm;
            } else {
                v = R[v];
                tl = tm + 1;
            }
        }
        return val[v];
    }
 
    ll updateArray(ll v, ll pos, ll el) {
        ll curv = cop(v);
        update(curv, 0, N, pos, el);
        return curv;
    }
 
    ll get(ll v, ll pos) {
        return query(v, 0, N, pos);
    }
};
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
Graphs find bridges
void dfs(int v, int p){
    s[v] = up[v] = timer++;
    for(auto vertex : g[v]){
        if (vertex == p) continue;
        if(s[vertex] == 0){
            dfs(vertex, v);
            up[v] = min(up[v], up[vertex]);
            if(up[vertex] > s[v]){
                isBridge[getId(vertex, v)] = 1;
            }
        } else {
            up[v] = min(up[v], s[vertex]);
        }
    }
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
Graphs find cut vertices
 
void dfs(int v, int p){
    int cnt = 0;
    s[v] = up[v] = timer ++;
    for(auto vertex : g[v]){
        if(s[vertex] == 0){
            cnt ++;
            dfs(vertex, v);
            up[v] = min(up[v], up[vertex]);
            if(up[vertex] >= s[v] && p != -1){
                artPoints.insert(v);
            }
        }
        else if(vertex != p){
            up[v] = min(up[v], s[vertex]);
        }
    }
    if(p == -1 && cnt > 1){
        artPoints.insert(v);
    }
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
LCA
 
const ll N = 1e5 + 10;
 
vll g[N];
ll tin[N], tout[N];
const ll logn = 20;
ll up[N][logn + 1];
ll timer = 1;
 
void dfs(ll v = 0, ll p = 0) {
    tin[v] = timer++;
    up[v][0] = p;
    rep(i, 1, logn + 1) {
        up[v][i] = up[up[v][i - 1]][i - 1];
    }
    for(auto i : g[v]) {
        if(i != p) {
            dfs(i, v);
        }
    }
    tout[v] = timer++;
}
 
bool parent(ll a, ll b) {
    return (tin[a] <= tin[b] && tout[a] >= tout[b]);
}
 
ll lca(ll a, ll b) {
    if(parent(a, b)) return a;
    if(parent(b, a)) return b;
    for(int i = logn; i >= 0; i--) {
        if(!parent(up[a][i], b)) a = up[a][i];
    }
    return up[a][0];
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
// Bor (trie)
// tested here : https://atcoder.jp/contests/code-festival-2016-qualb/submissions/18099350

const ll N = 4 * 1e5 + 5;
 
struct Node {
    ll cnt, flag;
    ll next[30];
};
 
vector<Node> bor = vector<Node>({Node()});
 
void add(string & t) {
    ll cur = 0;
    rep(i, 0, t.size()) {
        bor[cur].cnt++;
        if(bor[cur].next[t[i] - 'a'] == 0) {
            bor.pb(Node());
            bor[cur].next[t[i] - 'a'] = bor.size() - 1;
            cur = bor.size() - 1;
        } else {
            cur = bor[cur].next[t[i] - 'a'];
        }
    }
    bor[cur].flag = 1;
    bor[cur].cnt++;
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
 
long long prime(long long a){
 
    long long i;
 
    if (a==2)
        return 1;
    if(a<=1 || a%2==0)
        return 0;
 
   for(i=3;i*i<=a;i+=2){
        if(a%i==0)
            return 0;
   }
 
    return 1;
 
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
Improve Eratosfen sieve
 
const int N = 100;
int lp[N+1];
vector<int> pr;
 
for (int i=2; i<=N; ++i) {
    if (lp[i] == 0) {
        lp[i] = i;
        pr.push_back (i);
    }
for (int j=0; j<(int)pr.size() && pr[j]<=lp[i] && i*pr[j]<=N; ++j)
    lp[i * pr[j]] = pr[j];
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
 
long long gcd(long long a , long long b){
    while(b!=0){
        long long t = b;
        b = a % b;
        a = t;
    }
    return a;
}
 
long long lcm (long long a, long long b) {
    return a / gcd (a, b) * b;
}
 
int C (int n, int k) {
    double res = 1;
    for (int i=1; i<=k; ++i)
        res = res * (n-k+i) / i;
    return (int) (res + 0.01);
}
 
int CnkTr(){
    const int maxn = ...;
    int C[maxn+1][maxn+1];
    for (int n=0; n<=maxn; ++n) {
    C[n][0] = C[n][n] = 1;
    for (int k=1; k<n; ++k)
        C[n][k] = C[n-1][k-1] + C[n-1][k];
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
@Bogdan
Рюкзак:
 
for(int i=0;i<=w;i++){
    d[0][i]=0;
}
for(int i=1;i<=m;i++){
    for(int j=1;j<=w;j++){
            if(j-a[i]<0){
                d[i][j]=d[i-1][j];
            }
            else
        d[i][j]=max(a[i]+d[i-1][j-a[i]],d[i-1][j]);
    }
}
cout<<d[m][w];
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
bin search
 
while(l + 1 < r) {
    ll m = (l + r) / 2;
    if(can(m) < n) {
        r = m;
    } else {
        l = m;
    }
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
 
ll binpow (ll a, ll n, ll m) {
    ll res = 1;
    while (n) {
        if (n & 1)
            res = res * a % m;
        a = a * a % m;
        n >>= 1;
    }
    return res;
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
 
bool cmp(string findStr, string str) {
    ll n = findStr.size();
    findStr += "%" + str;
    vector<ll> p(findStr.size());
    ll k = 0;
    p[0] = 0;
    rep(i, 1, findStr.size()) {
        while(k > 0 && findStr[k] != findStr[i]) {
            k = p[k - 1];
        }
        if(findStr[k] == findStr[i]) {
            k++;
        }
        p[i] = k;
        if(p[i] == n) {
            //cout<<i - 2 * n<< " ";
            return 1;
        }
    }
    return 0;
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bohdan
// Dinic algorithm (finding maximum flow)
// https://official.contest.yandex.com/mw2020prefinals/contest/18054
 
const ll INF = 1e18;

struct edge {
    ll a, b, cap, flow;
};

struct Dinic {
    ll n, s, t;
    vll d, ptr, q, used;
    vector<edge> e;
    vector<vll> g;

    Dinic(ll sz, ll source, ll think) {
        n = sz;
        s = source;
        t = think;
        d.resize(sz, 0);
        ptr.resize(sz, 0);
        q.resize(sz, 0);
        used.resize(sz, 0);
        g.resize(sz);
    }

    void add_edge(ll a, ll b, ll cap) {
        edge e1 = {a, b, cap, 0};
        edge e2 = {b, a, 0, 0};
        g[a].push_back(e.size());
        e.push_back(e1);
        g[b].push_back(e.size());
        e.push_back(e2);
    }

    bool bfs() {
        ll qh = 0, qt = 0;
        q[qt++] = s;
        d.assign(n, -1);
        d[s] = 0;
        while (qh < qt && d[t] == -1) {
            ll v = q[qh++];
            for (size_t i=0; i < g[v].size(); ++i) {
                ll id = g[v][i], to = e[id].b;
                if (d[to] == -1 && e[id].flow < e[id].cap) {
                    q[qt++] = to;
                    d[to] = d[v] + 1;
                }
            }
        }
        return d[t] != -1;
    }

    ll dfs(ll v, ll flow) {
        if (!flow) return 0;
        if (v == t) return flow;
        for (; ptr[v] < g[v].size(); ++ptr[v]) {
            ll id = g[v][ptr[v]], to = e[id].b;
            if (d[to] != d[v] + 1) continue;
            ll pushed = dfs(to, min(flow, e[id].cap - e[id].flow));
            if (pushed) {
                e[id].flow += pushed;
                e[id^1].flow -= pushed;
                return pushed;
            }
        }
        return 0;
    }

    ll findMaxFlow() {
        ll flow = 0;
        for (;;) {
            if (!bfs())  break;
            ptr.assign(n, 0);
            while (ll pushed = dfs (s, INF)) flow += pushed;
        }
        return flow;
    }

    vll path;
    ll gl = 1;
    vvll decompose() {
        vvll ans;
        ll flow = dfsDecomp(s);
        while (flow) {
            gl++;
            reverse(path.begin(), path.end());
            ans.pb(path);
            path.clear();
            flow = dfsDecomp(s);
        }
        return ans;
    }

    ll dfsDecomp(ll v, ll curflow = INF) {
        if (curflow <= 0) return 0;
        if (v == t) return curflow;
        if (used[v] == gl) return 0;
        used[v] = gl;

        for (auto j : g[v]) {
            auto &i = e[j];
            if (i.flow) {
                ll cur = dfsDecomp(i.b, min(curflow, i.flow));
                if (cur) {
                    i.flow -= cur;
                    path.push_back(v + 1);
                    return cur;
                }
            }
        }
        return 0;
    }
};


////////////////////////////////////////////////

////////////////////////////////////////////////
@Bogdan
Non - recursive dfs
 
void dfs(ll v) {
 
    vector<pll> st;
    st.pb({v, -1});
    while(!st.empty()) {
        auto & cur = st.back();
        if(cur.second == -1) {
            res.pb(cur.first);
            used[cur.first] = 1;
        }
        ll in = cur.second + 1;
        cur.second = in;
        if(in < g[cur.first].size()) {
            ll i = g[cur.first][in];
            if(!used[i]) {
                st.pb({i, -1});
                continue;
            }
        } else {
            st.pop_back();
        }
    }
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bohdan
// Min cost max flow algorithm 
// Tested on Problem G from 6-th day Uzhgorod 2020
// Source: https://sites.google.com/site/indy256/algo_cpp/min_cost_flow

struct Edge {
    ll to, f, cap, cost, rev;
};

struct MinCostMaxFlow {
    ll n;
    vector<ll> prio, curflow, prevedge, prevnode, q, pot, inqueue;
    vector<vector<Edge>> graph;

    MinCostMaxFlow(ll cntNodes) {
        n = cntNodes;
        prio.resize(n, 0);
        curflow.resize(n, 0);
        prevedge.resize(n, 0);
        prevnode.resize(n, 0);
        q.resize(n, 0);
        pot.resize(n, 0);
        inqueue.resize(n, 0);
        graph.resize(n);
    }

    void addEdge(int s, int t, int cap, int cost) {
        Edge a = {t, 0, cap, cost, (ll)graph[t].size()};
        Edge b = {s, 0, 0, -cost, (ll)graph[s].size()};
        graph[s].push_back(a);
        graph[t].push_back(b);
    }

    void printGraph() {
        for (ll i = 0; i < n; ++i) {
            if (!graph[i].empty()) {
                cout << "from i : " << i << endl;
                for (auto e : graph[i]) {
                }
            }
        }
    }

    pll findMinCostFlow(ll s, ll t, ll maxf) {
        ll flow = 0;
        ll flowCost = 0;
        while (flow < maxf) {
            priority_queue<pll, vector<pll>, greater<pll> > q;
            q.push(make_pair(0LL, s));
            prio.assign(n, 1e18);

            prio[s] = 0;
            curflow[s] = 1e18;
            while (!q.empty()) {
                pll cur = q.top();
                ll d = cur.first;
                ll u = cur.second;
                q.pop();
                if (d != prio[u]) continue;

                for (ll i = 0; i < graph[u].size(); i++) {
                    Edge &e = graph[u][i];
                    ll v = e.to;
                    if (e.cap <= e.f) continue;
                    ll nprio = prio[u] + e.cost + pot[u] - pot[v];
                    if (prio[v] > nprio) {
                        prio[v] = nprio;
                        q.push(make_pair(nprio, v));
                        prevnode[v] = u;
                        prevedge[v] = i;
                        curflow[v] = std::min(curflow[u], e.cap - e.f);
                    }
                }
            }

            if (prio[t] == 1e18)
                break;

            for (ll i = 0; i < n; i++)
                pot[i] += prio[i];

            ll df = min(curflow[t], maxf - flow);
            flow += df;
            for (ll v = t; v != s; v = prevnode[v]) {
                Edge &e = graph[prevnode[v]][prevedge[v]];
                e.f += df;
                graph[v][e.rev].f -= df;
                flowCost += df * e.cost;
            }
        }
        return make_pair(flow, flowCost);
    }
};

////////////////////////////////////////////////

////////////////////////////////////////////////

@Bogdan
Fast read
 
inline ll read()
{
 char c=getchar();
 ll x=0,f=1;
 while(c>'9'||c<'0')
 {
  if(c=='-') f=-1;
  c=getchar();
 }
 while(c>='0'&&c<='9')
 {
  x=x*10+c-'0';
  c=getchar();
 }
 return x*f;
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
Queue with push_back, pop_front and find Min/Max O(1) complexity(average)
tested on https://atcoder.jp/contests/cf16-tournament-round3-open/tasks/asaporo_d

template <typename T, class F = function<T(const T&, const T&)>>
struct QueueWithOperation {
 
    F func;
    T defVal;
 
    QueueWithOperation(T def, const F& f) : func(f), defVal(def) {}
 
    stack<pair<T, T> > s1, s2;
 
    int size() {
        return s1.size() + s2.size();
    }
 
    bool isEmpty() {
        return size() == 0;
    }
 
    T getOperation() {
        if (isEmpty()) {
            return defVal;
        }
        if (!s1.empty() && !s2.empty()) {
            return func(s1.top().second, s2.top().second);
        }
        if (!s1.empty()) {
            return s1.top().second;
        }
        return s2.top().second;
    }
 
    void push(T val) {
        if (s2.empty()) {
            s2.push({val, val});
        } else {
            s2.push({val, func(val, s2.top().second)});
        }
    }
 
    void pop() {
        if (s1.empty()) {
            while (!s2.empty()) {
                if (s1.empty()) {
                    s1.push({s2.top().first, s2.top().first});
                } else {
                    s1.push({s2.top().first, func(s2.top().first, s1.top().second)});
                }
                s2.pop();
            }
        }
        assert(!s1.empty());
        s1.pop();
    }
};
 
// usage 
const ll K = 305;
vector<QueueWithOperation<ll> > dp(K, QueueWithOperation<ll>((ll)-1e18, [](ll i, ll j) { return max(i, j); }));
////////////////////////////////////////////////
 
Geometry
 
////////////////////////////////////////////////
@Bogdan
Psevdo multiplieng vectors
 
bool psevdo(pll a, pll b, pll c) {
    //return c under line ab or not
    return (((b.first - a.first) * (c.second - a.second) - (b.second - a.second) * (c.first - a.first)) >= 0);
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
Intersection line and circle
 
pair<pair<ld, ld>, pair<ld, ld>> intersection(ll x, ll y, ll x1, ll y1, ll xk, ll yk, ll r) {
    ll a1 = (y1 - y), b = (x - x1), c = -(x * (y1 - y) + y * (x - x1));
    ll c1 = a1 * xk + b * yk + c;
    ld x0 = xk - (ld)a1 * c1 / (ld)(a1 * a1 + b * b);
    ld y0 = yk - (ld)b * c1 / (ld)(a1 * a1 + b * b);
    if(c1 * c1 > r * r * (a1 * a1 + b * b)){
        return make_pair(make_pair(2e9, 2e9), make_pair(2e9, 2e9));
    }
    else if(c1 * c1 == r * r * (a1 * a1 + b * b)){
        return make_pair(make_pair(x0, y0), make_pair(x0, y0));
    }
    else {
        ld d = r * r - (ld)c1 * c1 / (ld)(a1 * a1 + b * b);
        ld mult = sqrt (d / (ld)(a1 * a1 + b * b));
        ld ax, ay, bx, by;
        ax = x0 + b * mult;
        bx = x0 - b * mult;
        ay = y0 - a1 * mult;
        by = y0 + a1 * mult;
        pair<ld, ld> cur1 = {ax, ay};
        pair<ld, ld> cur2 = {bx, by};
        return make_pair(cur1, cur2);
    }
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
Intersect two segments + (a, b, c in line)
 
bool peret(int x1,int y1,int x2,int y2,int  x3,int y3,int x4,int y4){
       double x ,y;
    int a=(y2-y1),b=(x1-x2),c=(x1*(y2-y1)+y1*(x1-x2)),a1=(y4-y3),b1=(x3-x4),c1=(x3*(y4-y3)+y3*(x3-x4));
    if((a*b1-a1*b)==0){
        return (intersect_1 (x1,x2,x3,x4) && intersect_1 (y1, y2,y3,y4));
    }
    else{
        x=(c1*b-b1*c)*1./(b*a1-b1*a)*1.;
        y=(c*a1-c1*a)*1./(b*a1-b1*a)*1.;
        if((min(x1,x2)<=x && x<=max(x1,x2)) && (min(y1,y2)<=y && y<=max(y1,y2))
        && (min(x3,x4)<=x && x<=max(x3,x4)) && (min(y3,y4)<=y && y <=max(y3,y4))){
            return 1;
        }
            return 0;
    }
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
Distance between segments
codeforces.com/gym/101243/attachments(problem I)
 
ld distance(ld xp1, ld yp1, ld xp2, ld yp2) {
    return sqrtl((xp1 - xp2) * (xp1 - xp2) + (yp1 - yp2) * (yp1 - yp2));
}
 
struct Segment {
    ll x1, y1, x2, y2;
 
    ld intersect(Segment seg) {
        ld dist1 = min(seg.measureDist(x1, y1), seg.measureDist(x2, y2));
        ld dist2 = min(measureDist(seg.x1, seg.y1), measureDist(seg.x2, seg.y2));
        return min(dist1, dist2);
    }
 
    ld measureDist(ll x, ll y) {
        ll a1 = (y2 - y1), b1 = (x1 - x2), c1 = -(x1 * a1 + y1 * b1);
        ll a2 = b1, b2 = -a1, c2 = (-a2 * x - b2 * y);
        ld xp = (ld)(-c2 * b1 + b2 * c1) / (ld)(b1 * a2 - b2 * a1);
        ld yp = (ld)(-c1 * a2 + c2 * a1) / (ld)(b1 * a2 - b2 * a1);
 
        ld dist;
 
        if (contains(xp, yp)) {
            dist = (ld)abs(a1 * x + b1 * y + c1) / (ld)sqrtl(a1 * a1 + b1 * b1);
        } else {
            dist = min(distance(x, y, x1, y1), distance(x, y, x2, y2));
        }
        return dist;
    }
 
    bool contains(ld x, ld y) {
        const ld eps = 1e-6;
        if(((ld)min(x1,x2) - eps <= x && x <= (ld)max(x1, x2) + eps) && ((ld)min(y1, y2) - eps <= y && y <= (ld)max(y1, y2) + eps)) {
            return true;
        } else {
            return false;
        }
    }
};
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Bogdan
 
Turn point
 
pair<ld, ld> turn(ld x, ld y, ld ug) {
    ld sn = sin(ug);
    ld cs = cos(ug);
 
    ld curx = mainx + (x - mainx) * cs - (y - mainy) * sn;
    ld cury = mainy + (x - mainx) * sn + (y - mainy) * cs;
    return {curx, cury};
}
////////////////////////////////////////////////
 
////////////////////////////////////////////////
@Dan
//Sparse Table
 
const ll N = 1e5 + 5;
const ll logN = log2(N);
 
ll mas[N];
 
ll table[logN + 1][N];
ll pow2[logN + 1];
 
void build() {
    pow2[0] = 1;
 
    rep(i, 1, logN + 1) {
        pow2[i] = pow2[i - 1];
        pow2[i] *= 2;
    }
 
    rep(i, 0, N)
           table[0][i] = mas[i];
 
    ll len = 2, power = 1, l = 0;
 
    while (power <= logN) {
        while (l + len - 1 < N) {
            table[power][l] = min(table[power - 1][l], table[power - 1][l + len / 2]);
            l++;
        }
 
        power++;
        len = pow2[power];
        l = 0;
    }
}
 
ll query(ll l, ll r) {
    if (l > r)
        swap(l, r);
 
    ll power = log2(r - l + 1);
 
    return min(table[power][l], table[power][r - pow2[power] + 1]);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
@Dan
// LIS
 
const ll N = 1e5 + 5;
 
ll mas[N], p[N], ind[N];
 
    vll dp(n + 1, mod);
 
    dp[0] = -mod;
 
    ll mx = 0;
 
    rep(i, 0, n) {
        ll j = upper_bound(all(dp), mas[i]) - dp.begin();
 
        if (dp[j - 1] < mas[i]) {
            dp[j] = mas[i];
            mx = max(mx, j);
 
            p[i] = ind[j - 1];
            ind[j] = i;
        }
    }
 
    ll curInd = ind[mx];
 
    deque <ll> d;
 
    rep(i, 0, mx) {
        d.pf(mas[curInd]);
 
        curInd = p[curInd];
    }
 
    cout << d.size() << "\n";
 
    for (auto c : d)
        cout << c << " ";
 
/////////////////////////////////////////////////////////////////////
@Dan
Hamilton cycle
 
const ll N = 21;
 
ll dp[(1 << N)][N];
 
rep(i, 0, n)
    rep(mask, 0, (1 << n))
        dp[mask][i] = mod;
 
dp[1][0] = 0;
 
rep(mask, 0, (1 << n))
    rep(i, 1, n)
        if ((mask >> i) & 1)
            rep(j, 0, n)
                if (i != j && (mask >> j) & 1)
                    dp[mask][i] = min(dp[mask][i], dp[mask ^ (1 << i)][j] + dst[i][j]);
 
ll mn = mod;
 
rep(i, 1, n)
    mn = min(mn, dp[(1 << n) - 1][i] + dst[0][i]);
 
return mn;
/////////////////////////////////////////////////////////////////////////////
@Dan
Дейкстра
 
const ll N = 1e3 + 5;
 
struct edge {
    ll v, dst;
};
 
const bool operator < (const edge &a, const edge &b) {
    return a.dst > b.dst;
}
 
vector <edge> graph[N];
 
    vll dst(n, mod);
 
    priority_queue <edge> q;
 
    q.push({beg, 0});
 
    dst[beg] = 0;
 
    while (!q.empty()) {
        edge cur = q.top();
 
        q.pop();
 
        if (cur.dst > dst[cur.v])
            continue;
 
        for (auto c : graph[cur.v]) {
            if (c.dst + cur.dst < dst[c.v]) {
                dst[c.v] = c.dst + cur.dst;
 
                q.push({c.v, dst[c.v]});
            }
        }
    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
@Dan
Fenwick
 
//one dimentional
 
void add(ll in, ll val) {
    for (ll i = in; i < N; i |= (i + 1))
        mas[i] += val;
}
 
ll sum(ll in) {
    ll s = 0;
 
    for (ll i = in; i >= 0; i = (i & (i + 1)) - 1)
        s += mas[i];
 
    return s;
}
 
// two dimentional
 
void add(ll x, ll y, ll val) {
    for (ll i = x; i < N; i |= (i + 1))
        for (ll j = y; j < N; j |= (j + 1))
            mas[i][j] += val;
}
 
ll sum(ll x, ll y) {
    ll s = 0;
 
    for (ll i = x; i >= 0; i = (i & (i + 1)) - 1)
        for (ll j = y; j >= 0; j = (j & (j + 1)) - 1)
            s += mas[i][j];
 
    return s;
}
 
////////////////////////////////////////////////////////////////////////////////////////////////////////
@Dan
// binpow and binmul + matrix
 
ll binmul(ll a, ll b, ll m) {
    ll res = 0;
 
    while (b) {
        if (b & 1)
            res = (res + a) % m;
 
        a <<= 1;
 
        b >>= 1;
 
        a %= m;
    }
 
    return res;
}
 
ll binpow(ll a, ll b, ll m) {
    ll res = 1;
 
    while (b) {
        if (b & 1)
            res = mul(res, a, m);
 
        a = mul(a, a, m);
 
        b >>= 1;
    }
 
    return res;
}
 
vvll mul(vvll a, vvll b) {
    ll N = a.size();
    ll K = b.size();
    ll M = b[0].size();
 
    vvll ans = vvll(N, vll(M, 0));
 
    rep(i, 0, N) {
        rep(j, 0, M) {
            rep(k, 0, K) {
                ans[i][j] += a[i][k] * b[k][j];
                ans[i][j] %= mod;
            }
        }
    }
 
    return ans;
}
 
vvll binpow(vvll a, ll b) {
    vvll ans = a;
    b--;
 
    while (b > 0) {
        if (b & 1) {
            ans = mul(ans, a);
        }
 
        a = mul(a, a);
 
        b >>= 1;
    }
 
    return ans;
}
 
////////////////////////////////////////////////////////////////////////////////////////////////////////
@Bogdan
 
Euler path
 
void dfs(ll v) {
    while(!g[v].empty()) {
        ll cur = g[v].back();
        g[v].pop_back();
        dfs(cur);
    }
 
    ans.pb(getS(v));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////
 
////////////////////////////////////////////////////////////////////////////////////////////////////////
@Bogdan
// z - function
 
vll zFunction (string & s) {
    ll n = s.size();
    vll z(n);
    ll l = 0, r = 0;
    rep(i, 1, n) {
        if (i <= r)
            z[i] = min (r - i + 1, z[i - l]);
        while (i + z[i] < n && s[z[i]] == s[i + z[i]])
            ++z[i];
        if (i + z[i] - 1 > r)
            l = i,  r = i + z[i] - 1;
    }
    return z;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////
@Dan
// Centroid decomposition
 
const ll N = 5e4 + 5;
 
vvll graph;
 
vll sz, vec;
 
ll cnt = 0, curSz = 0, k, n;
 
bool used[N];
 
map <ll, ll> m;
 
void countSz(ll a, ll p = -1) {
    sz[a] = 1;
 
    for (auto c : graph[a])
        if (c != p && !used[c]) {
            countSz(c, a);
 
            sz[a] += sz[c];
        }
}
 
ll findCenter(ll a, ll p = -1) {
    for (auto c : graph[a])
        if (!used[c] && c != p && sz[c] > curSz / 2)
            return findCenter(c, a);
 
    return a;
}
 
void count(ll a, ll h, ll p = -1) {
 
    vec.pb(h);
 
    cnt += m[k - h];
    //cout << "k - h" << " " << m[k - h] << "\n";
 
    for (auto c : graph[a])
        if (!used[c] && c != p)
            count(c, h + 1, a);
}
 
void rec(ll a) {
    m.clear();
 
    m[0] = 1;
 
    countSz(a);
 
    curSz = sz[a];
 
    ll center = findCenter(a);
 
    used[center] = 1;
 
    for (auto c : graph[center])
        if (!used[c]) {
            vec.clear();
            count(c, 1);
 
            for (auto cc : vec)
                m[cc]++;
        }
 
    for (auto c : graph[center])
        if (!used[c])
            rec(c);
}
 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
@Dan
// DSU
 
const ll N = 2e5 + 5;
 
ll p[N], s[N];
 
void add(ll i) {
    p[i] = i;
    s[i] = 1;
}
 
ll get(ll v) {
    return p[v] == v ? v : p[v] = get(p[v]);
}
 
void unite(ll a, ll b) {
    a = get(a);
    b = get(b);
 
    if (a != b) {
        if (s[a] > s[b])
            swap(a, b);
 
        p[a] = b;
        s[b] += s[a];
    }
}
 
//////////////////////////////////////////////////////////////////////////////////////////////////////
@Dan
// Knapsack with recovery
// 1 - index
 
rep(i, 1, n + 1)
        rep(j, 0, allW + 1) {
            dp[i][j] = dp[i - 1][j];
 
            if (j - w[i] >= 0)
                dp[i][j] = max(dp[i][j], dp[i - 1][j - w[i]] + p[i]);
        }
 
    ll cur = n, curW = allW;
 
    deque <ll> ans;
 
    while (dp[cur][curW]) {
        if (dp[cur - 1][curW] == dp[cur][curW])
            cur--;
        else {
            curW -= w[cur];
            ans.pf(cur);
            cur--;
        }
    }
 
    cout << dp[n][curW] << "\n";
////////////////////////////////////////////////////////////////////////////////////////////////////
 
////////////////////////////////////////////////////////////////////////////////////////////////////
@Bohdan
// new binary search
// (binary lifting)
 
for (ll i = 30; i >= 0; i--) {
        cur3 += (1LL << i);
        if (!can(cur3)) {
            cur3 -= (1LL << i);
        }
}
//////////////////////////////////////////////////////////////////////////////////////////////////
 
//////////////////////////////////////////////////////////////////////////////////////////////////
@Bohdan
// Matching - min vertex cover, max independent set, max matching (with Kuhn algorithm)
// tested on the problems from MPW official.contest.yandex.com/mw2020prefinals/contest/18051/enter
// ex. of AC solution: pastebin.com/hyv6yxaU
 
struct Matching {
    ll n, m;
    vvll g;
    vll lm, rm;
    ll gl = 0;
    vll used, usedCoverL, usedCoverR;
 
    Matching(ll leftComp, ll rightComp, vector<pll> & edges) {
        n = leftComp;
        m = rightComp;
        lm.assign(n, -1);
        rm.assign(m, -1);
        g.assign(n, vll({}));
        for (auto edge : edges) {
            g[edge.first].pb(edge.second);
        }
    }
 
    ll greedyInit() {
        ll cnt = 0;
        rep(i, 0, n) {
            for(auto j : g[i]) {
                if (lm[i] == -1 && rm[j] == -1) {
                    lm[i] = j;
                    rm[j] = i;
                    cnt++;
                }
            }
        }
        return cnt;
    }
 
    void clearUsed() {
        gl++;
    }
 
    ll dfs(ll v) {
        if (used[v] == gl) return 0;
        used[v] = gl;
 
        for (auto i : g[v]) {
            if (rm[i] == -1 || dfs(rm[i])) {
                lm[v] = i;
                rm[i] = v;
                return 1;
            }
        }
        return 0;
    }
 
    vector<pll> findMaxMatching() {
        lm.assign(n, -1);
        rm.assign(m, -1);
        used.assign(n, 0);
        gl = 0;
 
        ll matchingSize = greedyInit();
 
        clearUsed();
        rep(i, 0, n) {
            if (lm[i] == -1 && dfs(i)) {
                matchingSize++;
                clearUsed();
            }
        }
        clearUsed();
 
        vector<pll> matching;
        rep(i, 0, n) {
            if (lm[i] != -1) {
                matching.pb(make_pair(i, lm[i]));
            }
        }
 
        return matching;
    }
 
    void markCover(ll v) {
        if (usedCoverL[v]) return;
        usedCoverL[v] = 1;
        for(auto i : g[v]) {
            if (rm[i] != -1) {
                usedCoverR[i] = 1;
                markCover(rm[i]);
            }
        }
    }
 
    pair<vll, vll> findVertexCover() {
        vector<pll> matching = findMaxMatching();
        usedCoverL.assign(n, 0);
        usedCoverR.assign(m, 0);
 
        rep(i, 0, n) {
            if (lm[i] == -1) {
                markCover(i);
            }
        }
 
        vll left, right;
        rep(i, 0, n) {
            if (!usedCoverL[i]) left.pb(i);
        }
        rep(i, 0, m) {
            if (usedCoverR[i]) right.pb(i);
        }
 
        return make_pair(left, right);
    }
 
    pair<vll, vll> findIndependentSet() {
        auto cover = findVertexCover();
        vll left, right;
        vll curUsed(n, 0);
        for(auto i : cover.first) curUsed[i] = 1;
        rep(i, 0, n) {
            if (!curUsed[i]) {
                left.pb(i);
            }
        }
        curUsed.assign(m, 0);
        for (auto i : cover.second) curUsed[i] = 1;
        rep(i, 0, m) {
            if (!curUsed[i]) {
                right.pb(i);
            }
        }
        return make_pair(left, right);
    }
};
 
//////////////////////////////////////////////////////////////////////////////////////////////////
 
////////////////////////////////////////////////////////////////////////////////////////////////////
@Bohdan
// sparse table from tourist
 
template <typename T, class F = function<T(const T&, const T&)>>
class SparseTable {
 public:
  int n;
  vector<vector<T>> mat;
  F func;
 
  SparseTable(const vector<T>& a, const F& f) : func(f) {
    n = static_cast<int>(a.size());
    int max_log = 32 - __builtin_clz(n);
    mat.resize(max_log);
    mat[0] = a;
    for (int j = 1; j < max_log; j++) {
      mat[j].resize(n - (1 << j) + 1);
      for (int i = 0; i <= n - (1 << j); i++) {
        mat[j][i] = func(mat[j - 1][i], mat[j - 1][i + (1 << (j - 1))]);
      }
    }
  }
 
  T get(int from, int to) const {
    assert(0 <= from && from <= to && to <= n - 1);
    int lg = 32 - __builtin_clz(to - from + 1) - 1;
    return func(mat[lg][from], mat[lg][to - (1 << lg) + 1]);
  }
};
 
//////////////////////////////////////////////////////////////////////////////////////////////////
 
 
 
////////////////////////////////////////////////////////////////////////////////////////////////////
@Bohdan
// Fenwick tourist
 
template <typename T>
class fenwick {
 public:
  vector<T> fenw;
  int n;
 
  fenwick(int _n) : n(_n) {
    fenw.resize(n);
  }
 
  void modify(int x, T v) {
    while (x < n) {
      fenw[x] += v;
      x |= (x + 1);
    }
  }
 
  T get(int x) {
    T v{};
    while (x >= 0) {
      v += fenw[x];
      x = (x & (x + 1)) - 1;
    }
    return v;
  }
 
  T get(int l, int r) {
    return get(r) - (l - 1 >= 0 ? get(l - 1) : 0);
  }
};
 
//////////////////////////////////////////////////////////////////////////////////////////////////
 
//////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix struct (exponation, other operators)
// Tested on this problems:
// e-olymp.com/ru/problems/1482, e-olymp.com/ru/problems/1485
 
struct Matrix {
    vvll a;
    ll n, m;
    ll md = 1e18;
 
    Matrix(vvll cur) {
        if (cur.size() != 0) {
            a = cur;
            n = a.size();
            m = a[0].size();
        } else {
            n = 0, m = 0;
        }
    }
 
    Matrix(ll initn, ll initm) {
        n = initn;
        m = initm;
        a.assign(n, vll(m, 0));
    }
 
    Matrix operator * (const Matrix & cur) {
        if (m == cur.n) {
            Matrix res = Matrix(n, cur.m);
            rep(i, 0, n) {
                rep(j, 0, cur.m) {
                    rep(k, 0, m) {
                        res.a[i][j] += (a[i][k] * cur.a[k][j]) % md;
                        res.a[i][j] %= md;
                    }
                }
            }
            return res;
        }
        return Matrix(0, 0);
    }
 
    Matrix operator + (const Matrix & cur) {
        Matrix res = Matrix(min(n, cur.n), min(m, cur.m));
        rep(i, 0, min(n, cur.n)) {
            rep(j, 0, min(m, cur.m)) {
                res.a[i][j] = (a[i][j] + cur.a[i][j]) % md;
                res.a[i][j] %= md;
            }
        }
        return res;
    }
 
    void print() {
        cout << n << " " << m << endl;
        rep(i, 0, n) {
            rep(j, 0, m) {
                cout << a[i][j] << " ";
            }
            cout << endl;
        }
    }
 
    void read() {
        rep(i, 0, n) {
            rep(j, 0, m) {
                cin >> a[i][j];
            }
        }
    }
 
    Matrix binpow(Matrix cur, ll n) {
        Matrix res = Matrix(cur.n, cur.n);
        rep(i, 0, cur.n) {
            rep(j, 0, cur.n) {
                if (i == j) {
                    res.a[i][j] = 1;
                }
            }
        }
 
        while (n) {
            if (n & 1) res = res * cur;
            cur = cur * cur;
            n >>= 1;
        }
        return res;
    }
};
 
//////////////////////////////////////////////////////////////////////////////////////////////////
 
////////////////////////////////////////////////////////////////////////////////////////////////////
@Bohdan
// fact and revfact by modulo
https://c...content-available-to-author-only...s.com/contest/1312/submission/73322032
 
const ll N = 2 * 1e5 + 1;
ll md = 998244353;
 
ll fact[N], revfact[N];
 
ll cnk(ll n, ll k) {
    if (k == 0) return 1;
    if (n < k) return 0;
    return (1LL * fact[n] % md * revfact[n - k] % md * revfact[k] % md) % md;
}
 
ll binpow (ll a, ll n, ll m) {
    ll res = 1;
    while (n > 0) {
        if (n & 1)
            res = res * a % m;
        a = a * a % m;
        n >>= 1;
    }
    return res;
}
 
void precalc() {
    fact[0] = 1;
    fact[1] = 1;
    rep(i, 2, N) {
        fact[i] = (1LL * fact[i - 1] % md * i % md) % md;
    }
    revfact[0] = 1;
    revfact[N - 1] = binpow(fact[N - 1], md - 2, md);
    for(ll i = N - 2; i >= 1; i--) {
        revfact[i] = (1LL * revfact[i + 1] % md * (i + 1) % md) % md;
    }
}
 
//////////////////////////////////////////////////////////////////////////////////////////////////
 
////////////////////////////////////////////////////////////////////////////////////////////////////
@Bohdan
// cool treap on arrays
 
const ll SZ = 2e5 + 10;
 
ll cnt[SZ], left[SZ], right[SZ];
 
ll sum[SZ], val[SZ];
 
bool rev[SZ];
 
ll gl = 0;
 
ll newNode(ll cur) {
    gl++;
    cnt[gl] = 1;
    left[gl] = 0;
    right[gl] = 0;
    val[gl] = cur;
    sum[gl] = cur;
    return gl;
}
 
void update(ll t) {
    if (!t) return;
    cnt[t] = 1 + cnt[left[t]] + cnt[right[t]];
    sum[t] = val[t] + sum[left[t]] + sum[right[t]];
}
 
void push(ll t) {
    if (t && rev[t]) {
        rev[t] = 0;
        if (left[t]) rev[left[t]] ^= 1;
        if (right[t]) rev[right[t]] ^= 1;
        std::swap(left[t], right[t]);
    }
}
 
/*
void print(Node * t) {
    if (!t) return;
    print(t -> l);
    cout << t -> val << " " << t -> min << " " << t -> cnt << "   ";
    print(t -> r);
}
*/
 
std::pair<ll, ll> split(ll t, ll pos) {
    if (!t) {
        return std::make_pair(0, 0);
    } else {
        push(t);
        if (pos <= cnt[left[t]]) {
            auto cur = split(left[t], pos);
            left[t] = cur.second;
            update(t);
            return std::make_pair(cur.first, t);
        } else {
            auto cur = split(right[t], pos - cnt[left[t]] - 1);
            right[t] = cur.first;
            update(t);
            return std::make_pair(t, cur.second);
        }
    }
}
 
ll merge(ll l, ll r) {
    if (!l || !r) {
        return l ? l : r;
    } else {
        if ((rand_r(&SID) % (cnt[l] + cnt[r])) > cnt[r]) {
            push(l);
            right[l] = merge(right[l], r);
            update(l);
            return l;
        } else {
            push(r);
            left[r] = merge(l, left[r]);
            update(r);
            return r;
        }
    }
}
 
ll insert(ll t, ll cur, ll pos) {
    auto temp = split(t, pos);
    temp.first = merge(temp.first, cur);
    temp.first = merge(temp.first, temp.second);
    return temp.first;
}
 
std::pair<ll, ll> getSum(ll t, ll l, ll r) {
    auto temp = split(t, r);
    auto cur = split(temp.first, l - 1);
    ll res = sum[cur.second];
    ll ans = merge(cur.first, cur.second);
    ans = merge(ans, temp.second);
    return std::make_pair(ans, res);
}
 
ll revv(ll t, ll l, ll r) {
    auto temp = split(t, r);
    auto cur = split(temp.first, l - 1);
    if (cur.second) rev[cur.second] ^= 1;
    ll ans = merge(cur.first, cur.second);
    ans = merge(ans, temp.second);
    return ans;
}
 
//////////////////////////////////////////////////////////////////////////////////////////////////
 
 
////////////////////////////////////////////////////////////////////////////////////////////////////
@Bohdan
// some hashes
 
const ll N = 1e6 + 5;
ll md = 1e18 + 7;
ll steps[N];
ll hashes[N];
 
void precalcSteps() {
    ll k = 27;
    steps[0] = 1;
    rep(i, 1, N) {
        steps[i] = (steps[i - 1] * k) % md;
    }
}
 
void calcHashes(string & s) {
    rep(i, 0, s.size()) {
        hashes[i] = (s[i] - 'a' + 1) * steps[i];
        if (i) hashes[i] += hashes[i-1];
    }
}
 
ll checkPartHashes(ll i, ll j, ll len) {
    if (i > j) swap(i, j);
    ll hashI = hashes[i + len - 1];
    if (i)  hashI -= hashes[i-1];
    hashI = (hashI + md) % md;
    ll hashJ = hashes[j + len - 1];
    if (j)  hashJ -= hashes[j-1];
    hashJ = (hashJ + md) % md;
 
    return (hashI * steps[j - i] % md == hashJ);
}
 
ll calcHash(string s) {
    ll hsh = 0;
    rep(i, 0, s.size()) {
        hsh += ((s[i] - 'a' + 1) * steps[i]) % md;
        hsh %= md;
    }
    return hsh;
}
 
//////////////////////////////////////////////////////////////////////////////////////////////////
 
////////////////////////////////////////////////////////////////////////////////////////////////////
@Bohdan
// suffix array (with lcp)
// https://c...content-available-to-author-only...s.com/edu/course/2/lesson/2/4/practice/contest/269119/problem/A
 
vll buildEkv(vector<pair<pll, ll> > & a) {
    vll ekv(a.size());
    ll cnt = 0;
    ekv[a.front().second] = 0;
    rep(i, 1, a.size()) {
        if (a[i].first != a[i - 1].first) {
            ekv[a[i].second] = ++cnt;
        } else {
            ekv[a[i].second] = cnt;
        }
    }
    return ekv;
}
 
void countSortByFirst(vector<pair<pll, ll> > & cur, ll n) {
    vll cnt(n, 0);
    rep(i, 0, cur.size()) cnt[cur[i].first.first]++;
    vll pos(n);
    pos[0] = 0;
    rep(i, 1, n) pos[i] = pos[i - 1] + cnt[i - 1];
 
    vector<pair<pll, ll>> newcur(cur.size());
    for(auto i : cur) {
        newcur[pos[i.first.first]] = i;
        pos[i.first.first]++;
    }
    cur.swap(newcur);
}
 
string s;
vector<pair<pll, ll> > cur;
 
int main() {
    // ios;
    cin >> s;
    s += "$";
    ll n = s.size();
    vll ekv;
    rep(i, 0, n) {
        cur.pb(make_pair(make_pair(s[i], s[i]), i));
    }
    countSortByFirst(cur, 256);
    ekv = buildEkv(cur);
    ll k = 0;
 
    while((1 << k) < n) {
        auto prevcur = cur;
        rep(i, 0, n) {
            ll stIndex = (cur[i].second - (1 << k) + n) % n;
            cur[i] = make_pair(make_pair(ekv[stIndex], ekv[cur[i].second]), stIndex);
        }
        countSortByFirst(cur, n);
        ekv = buildEkv(cur);
        k++;
    }
 
    rep(i, 0, n) {
        cout << cur[i].second << " ";
    }
    cout << endl;
 
    vll lcp(n);
    k = 0;
    rep(i, 0, n - 1) {
        ll posi = ekv[i];
        ll j = cur[posi - 1].second;
        while(s[i + k] == s[j + k]) k++;
        lcp[posi] = k;
        k = max(k - 1, 0LL);
    }
    rep(i, 1, lcp.size()) cout << lcp[i] << " ";
    cout << endl;
 
    return 0;
}
 
//////////////////////////////////////////////////////////////////////////////////////////////////
 
//////////////////////////////////////////////////////////////////////////////////////////////////
// Fraction structure from Bohdan

struct Fraction {
    ll num, denum;
 
    Fraction (ll newNum, ll newDenum) {
        ll sign = (newNum * newDenum > 0);
        num = newNum / __gcd(newNum, newDenum);
        denum = newDenum / __gcd(newNum, newDenum);
        if ((num * denum > 0) != (newNum * newDenum > 0)) {
            num *= -1;
        } 
    }

    bool operator < (const Fraction & right) const {
        return num * right.denum < denum * right.num;
    }

    Fraction operator / (const Fraction & right) {
        return Fraction(num * right.denum, denum * right.num);
    }

    Fraction operator - (const Fraction & right) {
        return Fraction(num * right.denum - right.num * denum, denum * right.denum);
    }

    Fraction operator + (const Fraction & right) {
        return Fraction(num * right.denum + right.num * denum, denum * right.denum);
    }

    Fraction operator * (const Fraction & right) {
        return Fraction(num * right.num, denum * right.denum);
    } 

    bool operator == (const Fraction & right) const {
        return num == right.num && denum == right.denum;
    }
};
//////////////////////////////////////////////////////////////////////////////////////////////////
 
//////////////////////////////////////////////////////////////////////////////////////////////////
// @Bohdan
// Heavy - light decomposition(HLD)
// from homework1(semestr 2) - yandex
// https://c...content-available-to-author-only...x.ru/contest/17428/problems/E/
 
vll parent, depth, heavy, head, pos;
ll cur_pos;
 
ll n;
vvll g;
 
vll tr;
 
ll findMax(ll v, ll tl, ll tright, ll l, ll r) {
    if (l > r) return 0;
    if (l == tl && r == tright) {
        return tr[v];
    }
    ll tm = (tl + tright) / 2;
    return max(findMax(v * 2 + 1, tl, tm, l, min(r, tm)), findMax(v * 2 + 2, tm + 1, tright, max(tm + 1, l), r));
}
 
void addValue(ll v, ll l, ll r, ll pos, ll val) {
    if (l == r) {
        tr[v] += val;
        return;
    }
    ll m = (l + r) / 2;
    if (pos <= m) {
        addValue(v * 2 + 1, l, m, pos, val);
    } else {
        addValue(v * 2 + 2, m + 1, r, pos, val);
    }
    tr[v] = max(tr[v * 2 + 1], tr[v * 2 + 2]);
}
 
ll segment_tree_query(ll a, ll b) {
    return findMax(0, 0, n - 1, a, b);
}
 
ll query(ll a, ll b) {
    ll res = 0;
 
    for (; head[a] != head[b]; b = parent[head[b]]) {
        if (depth[head[a]] > depth[head[b]])
            swap(a, b);
        ll cur_heavy_path_max = segment_tree_query(pos[head[b]], pos[b]);
        res = max(res, cur_heavy_path_max);
    }
    if (depth[a] > depth[b])
        swap(a, b);
 
    ll last_heavy_path_max = segment_tree_query(pos[a], pos[b]);
    res = max(res, last_heavy_path_max);
    return res;
}
 
ll dfs(ll v) {
    ll sz = 1;
    ll max_c_size = 0;
    for (ll c : g[v]) {
        if (c != parent[v]) {
            parent[c] = v, depth[c] = depth[v] + 1;
            ll c_size = dfs(c);
            sz += c_size;
            if (c_size > max_c_size)
                max_c_size = c_size, heavy[v] = c;
        }
    }
    return sz;
}
 
void decompose(ll v, ll h) {
    head[v] = h, pos[v] = cur_pos++;
    if (heavy[v] != -1)
        decompose(heavy[v], h);
    for (ll c : g[v]) {
        if (c != parent[v] && c != heavy[v])
            decompose(c, c);
    }
}
 
void init() {
    parent.resize(n);
    depth.resize(n);
    heavy.resize(n, -1);
    head.resize(n);
    pos.resize(n);
    tr.resize(4 * n, 0);
    cur_pos = 0;
 
    dfs(0);
    decompose(0, 0);
}
 
int main() {
    ios;
    cin >> n;
    g.resize(n);
    rep(i, 0, n - 1) {
        ll a, b;
        cin >> a >> b;
        g[a - 1].pb(b - 1);
        g[b - 1].pb(a - 1);
    }
 
    init();
 
    ll m;
    cin >> m;
    while(m--) {
        char q;
        cin >> q;
        ll a, b;
        cin >> a >> b;
        if (q == 'I') {
            addValue(0, 0, n - 1, pos[a - 1], b);
        } else {
            cout << query(a - 1, b - 1) << endl;
        }
    }
 
    return 0;
}
 
//////////////////////////////////////////////////////////////////////////////////////////////////
 
 
//////////////////////////////////////////////////////////////////////////////////////////////////
// Aho - Corasick (@Bohdan)
// from e-olymp.com/ru/contests/4289/problems/34250
// implementation from habr.com/ru/post/198682/
 
const ll sz = 300;
char minSymbol = char(32);
 
struct Node {
    ll next[sz], pat_num, suff_link, move[sz], par, suff_flink;
    bool flag;
    char symb;
};
 
vector<Node> bohr;
vector<string> pattern;
 
Node makeNode(ll p, char c){
    Node v;
    memset(v.next, 255, sizeof(v.next));
    memset(v.move, 255, sizeof(v.move));
 
    v.flag = false;
    v.suff_link = -1;
    v.par = p;
    v.symb = c;
    v.suff_flink = -1;
    return v;
}
 
void bohr_ini() {
    bohr.push_back(makeNode(0, static_cast<char>(0)));
}
 
void add_string_to_bohr(const string& s) {
    ll num = 0;
    rep(i, 0, s.size()) {
        char ch = s[i] - minSymbol;
        if (bohr[num].next[ch] == -1){
            bohr.push_back(makeNode(num, ch));
            bohr[num].next[ch] = bohr.size() - 1;
        }
        num = bohr[num].next[ch];
    }
    bohr[num].flag = true;
    pattern.push_back(s);
    bohr[num].pat_num = pattern.size() - 1;
}
 
ll get_move(ll v, ll ch);
 
ll get_suff_link(ll v){
    if (bohr[v].suff_link == -1) {
        if (v == 0 || bohr[v].par == 0) {
            bohr[v].suff_link = 0;
        } else {
            bohr[v].suff_link =
            get_move(get_suff_link(bohr[v].par), bohr[v].symb);
        }
    }
    return bohr[v].suff_link;
}
 
ll get_move(ll v, ll ch){
    if (bohr[v].move[ch] == -1) {
        if (bohr[v].next[ch] != -1) {
            bohr[v].move[ch] = bohr[v].next[ch];
        } else if (v == 0) {
            bohr[v].move[ch] = 0;
        } else {
            bohr[v].move[ch] = get_move(get_suff_link(v), ch);
        }
    }
    return bohr[v].move[ch];
}
 
ll get_suff_flink(ll v) {
    if (bohr[v].suff_flink == -1) {
        ll u = get_suff_link(v);
        if (u == 0) {
            bohr[v].suff_flink = 0;
        } else {
            bohr[v].suff_flink = (bohr[u].flag) ? u : get_suff_flink(u);
        }
    }
    return bohr[v].suff_flink;
}
 
ll check(int v, int i){
    for (ll u = v; u != 0; u = get_suff_flink(u)){
        if (bohr[u].flag) return 1;
    }
    return 0;
}
 
ll find_all_pos(const string& s){
    ll u = 0;
    rep(i, 0, s.size()) {
        u = get_move(u, s[i] - minSymbol);
        if (check(u, i + 1)) return 1;
    }
    return 0;
}
 
//////////////////////////////////////////////////////////////////////////////////////////////////
 
 
//////////////////////////////////////////////////////////////////////////////////////////////////
// Link - cut tree
// Uzhorod school 2 day (Tsypko) (A + B + C)
// olymp.uzhnu.edu.ua/cgi-bin/new-client?contest_id=312
 
struct Node {
    ll left = 0, right = 0, parent = 0, link = 0, sz = 1, rev = 0;
    Node() : left(0), right(0), parent(0), link(0), sz(1), rev(0) {}
};
 
struct LinkCutTree {
    vector<Node> tr;
 
    LinkCutTree(ll size) {
        tr.resize(size);
        tr[0].sz = 0;
    }
 
    void recomp(ll v) {
        tr[v].sz = 1 + tr[tr[v].left].sz + tr[tr[v].right].sz;
    }
 
    void set_parent(ll child, ll parent) {
        if (child) {
            tr[child].parent = parent;
        }
    }
 
    void revers(ll v) {
        if (v) {
            tr[v].rev ^= 1;
        }
    }
 
    void push(ll v) {
        if (v && tr[v].rev) {
            swap(tr[v].left, tr[v].right);
            tr[v].rev = 0;
            revers(tr[v].left);
            revers(tr[v].right);
        }
    }
 
    void keep_parent(ll v) {
        set_parent(tr[v].left, v);
        set_parent(tr[v].right, v);
        recomp(v);
    }
 
    void rotation(ll parent, ll child) {
        ll gparent = tr[parent].parent;
        if (gparent) {
            if (tr[gparent].left == parent) {
                tr[gparent].left = child;
            }
            else {
                tr[gparent].right = child;
            }
        }
 
        if(tr[parent].left == child) {
            tr[parent].left = tr[child].right;
            tr[child].right = parent;
        }
        else {
            tr[parent].right = tr[child].left;
            tr[child].left = parent;
        }
        keep_parent(parent);
        keep_parent(child);
        tr[child].parent = gparent;
    }
 
    ll splay(ll v) {
        if (!tr[v].parent) return v;
        ll parent = tr[v].parent;
        ll gparent = tr[parent].parent;
        if (!gparent) {
            rotation(parent, v);
            return v;
        } else {
            ll zigzig = ((tr[gparent].left == parent) == (tr[parent].left == v));
            if (zigzig) {
                rotation(gparent, parent);
                rotation(parent, v);
            } else {
                rotation(parent, v);
                rotation(gparent, v);
            }
        }
        return splay(v);
    }
 
    void cleanup(ll v) {
        if(!v) return;
        cleanup(tr[v].parent);
        push(v);
    }
 
    ll supportRoot(ll v) {
        cleanup(v);
        return splay(v);
    }
 
    ll found(ll v, ll lhs) {
        if (!v) return 0;
        push(v);
        ll threshold = tr[tr[v].left].sz;
        if (lhs == threshold) return splay(v);
        if (lhs < threshold && tr[v].left) return found(tr[v].left, lhs);
        if (lhs > threshold && tr[v].right) return found(tr[v].right, lhs - threshold - 1);
        return splay(v);
    }
 
    pair<ll, ll> split(ll root, ll lhs) {
        if (!root) return make_pair(0, 0);
        root = found(root, lhs);
        if (tr[root].sz <= lhs) return make_pair(root, 0);
 
        set_parent(tr[root].left, 0);
        tr[root].left = 0;
        recomp(root);
        return make_pair(tr[root].left, root);
    }
 
    ll merg(ll left, ll right) {
        if (!right) return left;
        if (!left) return right;
        right = found(right, 0);
        tr[right].left = left;
        keep_parent(right);
        return supportRoot(right);
    }
 
    ll cutout(ll v) {
        supportRoot(v);
        auto cur = split(v, tr[tr[v].left].sz + 1);
        if (cur.second) tr[found(supportRoot(cur.second), 0)].link = v;
        return v;
    }
 
    ll expose(ll v) {
        v = found(cutout(v), 0);
        while (tr[v].link) {
            ll nxt = cutout(tr[v].link);
            tr[v].link = 0;
            v = found(merg(nxt, v), 0);
        }
        return v;
    }
 
    void cut(ll child, ll parent) {
        expose(parent);
        tr[child].link = 0;
    }
 
    void cut(ll v) {
        expose(v);
        supportRoot(v);
        if (tr[v].left) {
            ll p = found(v, tr[tr[v].left].sz - 1);
            cut(v, p);
        }
    }
 
    ll depth(ll v) {
        expose(v);
        return tr[supportRoot(v)].sz - 1;
    }
 
    ll lca(ll u, ll v) {
        expose(u);
        if (found(supportRoot(v), 0) == found(supportRoot(u), 0)) return v;
        expose(v);
        if (found(supportRoot(v), 0) == found(supportRoot(u), 0)) return u;
 
        return tr[found(supportRoot(u), 0)].link;
    }
 
    ll link(ll child, ll parent) {
        tr[child].link = parent;
        return expose(child);
    }
 
    ll find_root(ll v) {
        return expose(v);
    }
 
    void revert(ll v) {
        ll root = expose(v);
        revers(supportRoot(root));
    }
 
    ll dist(ll u, ll v) {
        revert(u);
        ll w = expose(v);
        if (w != u) return -1;
        return depth(v);
    }
 
    void add(ll u, ll v) {
        revert(u);
        link(u, v);
    }
 
    void rem(ll u, ll v) {
        revert(v);
        cut(u, v);
    }
};
//////////////////////////////////////////////////////////////////////////////////////////////////
@Den
// hashes
 
const ll N = 1e6 + 5;
const ll P = 79;
 
vll prefHash;
vll sufHash;
ll powP[N];
ll n;
string s;
 
ll getPrefHash(ll i, ll j) {
    ll hash = prefHash[j];
 
    if (i) {
        hash -= prefHash[i - 1];
        hash += mod;
    }
 
    return hash;
}
 
ll getSufHash(ll i, ll j) {
    ll hash = sufHash[i];
 
    if (j != n - 1) {
        hash -= sufHash[j + 1];
        hash += mod;
    }
 
    return hash;
}
 
bool checkPal(ll i, ll j) {
    ll hash1 = getPrefHash(i, j);
 
    // rev hash
    ll hash2 = getSufHash(i, j);
 
    hash1 *= powP[n - j - 1];
    hash2 *= powP[i];
 
    hash1 %= mod;
    hash2 %= mod;
 
    return hash1 == hash2;
}
 
void countHashes() {
    rep(i, 0, n) {
        if (i)
            prefHash[i] += prefHash[i - 1];
 
        prefHash[i] %= mod;
 
        prefHash[i] += powP[i] * (s[i] - 'a' + 1) % mod;
        prefHash[i] %= mod;
    }
 
    per(i, n, 0) {
        if (i != n - 1)
            sufHash[i] += sufHash[i + 1];
 
        sufHash[i] %= mod;
 
        sufHash[i] += powP[n - i - 1] * (s[i] - 'a' + 1) % mod;
        sufHash[i] %= mod;
    }
}
 
int main() {
    //ios
    //mt19937 rng(chrono::steady_clock::now().time_since_epoch().count())
 
    powP[0] = 1;
 
    rep(i, 1, N) {
        powP[i] = powP[i - 1] * P % mod;
    }
 
    cin >> s;
 
    n = s.size();
 
    prefHash.assign(n, 0);
    sufHash.assign(n, 0);
 
    countHashes();
 
    return 0;
}
//////////////////////////////////////////////////////////////////
@Den
// Geometry
 
const ld pi = 3.1415926;
 
struct pt {
    ld x, y;
 
    bool scan() {
        if (cin >> x >> y)
            return true;
        return false;
    }
    bool onSegment(pt a, pt b) { // works if we know that pt is on line builded on ab
            return min(a.x, b.x) <= x && x <= max(b.x, a.x) &&
                min(a.y, b.y) <= y && y <= max(b.y, a.y);
    }
};
 
struct Line {
    ld a, b, c;
};
 
ld sqr(ld a) {
    return a * a;
}
 
ld distance(pt a, pt b) {
    return sqrtl(sqr(a.x - b.x) + sqr(a.y - b.y));
}
 
ld vec(pt a, pt b, pt c) {
    return (a.x - b.x) * (c.y - b.y) - (c.x - b.x) * (a.y - b.y);
}
 
ld ccw(pt a, pt b, pt c) {
    return vec(a, b, c) > 0;
}
 
ld cw(pt a, pt b, pt c) {
    return vec(a, b, c) < 0;
}
 
ld det(ld a, ld b, ld c, ld d) {
    // a b
    // c d
    return a * d - b * c;
}
 
Line makeLine(pt a, pt b) {
    ld n = a.x - b.x;
    ld m = a.y - b.y;
 
    return {m, -n, -(b.x * m - n * b.y)};
}
 
Line bisect(pt a, pt b, pt c) {
    ld phi = dst(b, c) / dst(a, b);
 
    pt k = {(c.x + a.x * phi) / (1 + phi),
        (c.y + a.y * phi) / (1 + phi)};
 
    return makeLine(b, k);
}
 
pt intersection(Line l1, Line l2) {
    l1.c *= -1;
    l2.c *= -1;
 
    ld delta = det(l1.a, l1.b, l2.a, l2.b);
    ld deltaX = det(l1.c, l1.b, l2.c, l2.b);
    ld deltaY = det(l1.a, l1.c, l2.a, l2.c);
 
    return {deltaX / delta, deltaY / delta};
}
 
ld dstFromPtToLine(Line l, pt p) {
    return abs(l.a * p.x + l.b * p.y + l.c) / sqrt(sqr(l.a) + sqr(l.b));
}

struct Rect {
    pt a, b, c, d;
 
    bool checkInside(pt cur) { // vertexes must be sorted clockwise or counter-clockwise
        ld v1 = vec(a, b, cur);
        ld v2 = vec(b, c, cur);
        ld v3 = vec(c, d, cur);
        ld v4 = vec(d, a, cur);
 
        ll cnt1 = (v1 >= 0) + (v2 >= 0) + (v3 >= 0) + (v4 >= 0);
        ll cnt2 = (v1 <= 0) + (v2 <= 0) + (v3 <= 0) + (v4 <= 0);
 
        return cnt2 == 4 || cnt1 == 4;
    }
};
 
pt turn(pt p, pt main, ld degree) { // degree in radians, counter-clockwise direction
    ld sn = sin(degree);
    ld cs = cos(degree);
 
    ld curx = main.x + (p.x - main.x) * cs - (p.y - main.y) * sn;
    ld cury = main.y + (p.x - main.x) * sn + (p.y - main.y) * cs;
    return {curx, cury};
}
 
pt mid(pt a, pt b) {
    return {(a.x + b.x) / 2, (a.y + b.y) / 2};
}
 
ld findArea(vector <pt> &poly) {
    ld ans = 0;
    rep(i, 0, poly.size()) {
        pt
            p1 = i ? poly[i - 1] : poly.back(),
            p2 = poly[i];
        ans += (p1.x - p2.x) * (p1.y + p2.y);
    }
    return abs(ans) / 2;
}

// Rect intersection
// tested on https://codeforces.com/contest/1216/problem/C

struct Rect {
    pt a, b;
    
    void scan() {
        a.scan();
        b.scan();
    }
    
    ll getS() {
        return abs(a.x - b.x) * abs(a.y - b.y);
    }
};

Rect intersection(Rect a, Rect b) {
    ll l = max(min(a.a.x, a.b.x), min(b.a.x, b.b.x));
    ll r = min(max(a.a.x, a.b.x), max(b.a.x, b.b.x));
    
    ll d = max(min(a.a.y, a.b.y), min(b.a.y, b.b.y));
    ll u = min(max(a.a.y, a.b.y), max(b.a.y, b.b.y));
    
    if (l >= r || d >= u) {
        return Rect{{0, 0}, {0, 0}};
    }
    
    return Rect {{l, d}, {r, u}};
}

///////////////////////////////////////////////////////////
@Den
// bit trie
const ll bits = 32;
 
struct Vertex {
    ll next[2];
 
    Vertex() {
        next[0] = next[1] = -1;
    }
};
 
vector <Vertex> trie(1);
 
void addToTrie(ll a, vector <Vertex> &trie) {
    ll it = 0;
 
    if (trie.empty()) {
        trie.pb(Vertex());
    }
 
    per(j, bits, 0) {
        bool bit = (a & (1ll << j));
 
        if (trie[it].next[bit] == -1) {
            trie[it].next[bit] = trie.size();
            trie.pb(Vertex());
        }
 
        it = trie[it].next[bit];
    }
}
 
ll findMin(ll x, vector <Vertex> &trie) {
    ll it = 0;
    ll ans = 0;
 
    if (trie.empty()) {
        trie.pb(Vertex());
    }
 
    per(j, bits, 0) {
        bool needBit = (x & (1ll << j));
 
        if (trie[it].next[needBit] == -1) {
            needBit = !needBit;
        }
 
        if (needBit)
            ans += (1ll << j);
 
        it = trie[it].next[needBit];
    }
    return ans;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
@Den
// Convex hull trick

struct line {
    ll k, b;
 
    line(ll k, ll b) : k(k), b(b) {}
    ll get(ll x) {
        if (x == -inf)
            return -inf;
        return x * k + b;
    }
};
 
vector <line> lines;
vll dots;
 
ll cross(line a, line b) {
    ll x = floor((b.b - a.b) / (a.k - b.k));
    return x;
}

void addLine(line cur) {
    while (lines.size() && lines.back().get(dots.back()) < cur.get(dots.back())) { // for max val
        lines.pop_back();
        dots.pop_back();
    }
 
    if (lines.empty()) {
        dots.pb(-inf);
    } else {
        dots.pb(cross(lines.back(), cur));
    }
 
    lines.pb(cur);
}
 
ll getMax(ll x) {
    if (lines.empty()) return -inf;
    ll pos = lower_bound(all(dots), x) - dots.begin() - 1;
    return lines[pos].get(x);
}
 
////////////////////////////////////////////////////////
@Dan
// Segment Tree

struct SegmentTree {
private:
    vll t;
    vll toAdd;
    ll n;
 
    ll merge(ll a, ll b) {
        return min(a, b);
    }
 
    void pull(ll it) {
        t[it] = merge(t[it * 2 + 1], t[it * 2 + 2]);
    }

    void build(ll l, ll r, ll it) {
        if (l == r) {
            t[it] = a[l];
            return;
        }
        
        ll m = (l + r) / 2;
        
        build(l, m, it * 2 + 1);
        build(m + 1, r, it * 2 + 2);
        
        pull(it);
    }
 
    void push(ll v) {
        toAdd[v * 2 + 1] += toAdd[v];
        toAdd[v * 2 + 2] += toAdd[v];
 
        t[v * 2 + 1] += toAdd[v];
        t[v * 2 + 2] += toAdd[v];
 
        toAdd[v] = 0;
    }

    ll query(ll l, ll r, ll ql, ll qr, ll it) {
        if (ql > qr)
            return -inf;
        if (l == ql && r == qr) {
            return t[it];
        }
 
        ll m = (l + r) / 2;
 
        push(it);
 
        return merge(
                   query(l, m, ql, min(qr, m), it * 2 + 1),
                   query(m + 1, r, max(m + 1, ql), qr, it * 2 + 2));
    }
 
    void change(ll l, ll r, ll ql, ll qr, ll val, ll it) {
        if (ql > qr)
            return;
        if (l == ql && r == qr) {
            t[it] += val;
            toAdd[it] += val;
            return;
        }
 
        ll m = (l + r) / 2;
 
        push(it);
 
        change(l, m, ql, min(qr, m), val, it * 2 + 1);
        change(m + 1, r, max(m + 1, ql), qr, val, it * 2 + 2);
 
        pull(it);
    }
public:
   SegmentTree(ll _n) {
        n = _n;
        t.resize(n * 4, 0);
        toAdd.resize(n * 4, 0);
    }
 
    void change(ll l, ll r, ll val) {
        if (l > r)
            swap(l, r);
        change(0, n - 1, l, r, val, 0);
    }
 
    ll query(ll l, ll r) {
        if (l > r)
            swap(l, r);
        return query(0, n - 1, l, r, 0);
    }

    void build() {
        build(0, n - 1, 0);
    }
};
///////////////////////////////////////////////
@Dan
// Heavy Light Decomposition
 
auto segmentTree = SegmentTree();
 
ll in[N];
ll out[N];
ll nxt[N];
ll par[N];
ll h[N];
ll sz[N];
 
vll graph[N];
 
ll query(ll a, ll b) {
    ll ans = -inf;
 
    while (nxt[a] != nxt[b]) {
        if (h[nxt[a]] > h[nxt[b]])
            swap(a, b);
 
        ans = max(ans, segmentTree.query(in[nxt[b]], in[b]));
 
        b = par[nxt[b]];
    }
 
    ans = max(ans, segmentTree.query(in[a], in[b]));
 
    return ans;
}
 
void dfs(ll a, ll p = -1, ll depth = 0) {
    sz[a] = 1;
    h[a] = depth;
    par[a] = p;
 
    for (auto &c : graph[a])
        if (c != p) {
            dfs(c, a, depth + 1);
 
            sz[a] += sz[c];
 
            if (sz[graph[a][0]] < sz[c]) {
                swap(graph[a][0], c);
            }
        }
}
 
ll timer = -1;

void decompose(ll a, ll p = -1) {
    in[a] = ++timer;
 
    for (auto c : graph[a])
        if (c != p) {
            nxt[c] = (c == graph[a][0] ? nxt[a] : c);
            decompose(c, a);
        }
 
    out[a] = timer;
}
 
void initHLD() {
    dfs(0);
    decompose(0);
}
 
void solve() {
    // don't forget about ios
    // ll n;
    cin >> n;
 
    rep(i, 0, n - 1) {
        ll a, b;
        cin >> a >> b;
        a--;
        b--;
 
        graph[a].pb(b);
        graph[b].pb(a);
    }
 
    initHLD();
 
    ll m;
    cin >> m;
    rep(i, 0, m) {
        string type;
        cin >> type;
        if (type == "add") {  // add to subtree
            ll a, val;
            cin >> a >> val;
            a--;
            segmentTree.add(in[a], out[a], val);
        } else if (type == "max"){  // max on path
            ll a, b;
            cin >> a >> b;
            a--;
            b--;
            cout << query(a, b) << "\n";
        }
    }
}
/////////////////////////////////////////////////////////////////////////
@Dan
// block cut tree
// Tested on Uzhhorod 2020. Day 2. Tsypko

vll graph[N];
ll fup[N];
ll timer = 0;
ll tin[N];
ll used[N];

vll graph_bct[N];
ll isArt[N];

vll vec;
ll n;

void addComp(vll &v) {
    for (auto c : v) {
        graph_bct[c].pb(n);
        graph_bct[n].pb(c);
    }
    
    n++;
}


void dfs_bct(ll a = 0, ll p = -1) {
    used[a] = 1;
    
    fup[a] = tin[a] = ++timer;
    vec.pb(a);
    
    for (auto c : graph[a]) {
        if (c != p) {
            if (!used[c]) {
                dfs_bct(c);
                
                fup[a] = min(fup[a], fup[c]);
                
                if (fup[c] >= tin[a]) { // a - cut vertex
                    isArt[a] = 1;
                    vll comp;
                    
                    comp.pb(a);
                    while (comp.back() != c) {
                        comp.pb(vec.back());
                        vec.pop_back();
                    }
                    
                    addComp(comp);
                }
            } else {
                fup[a] = min(fup[a], tin[c]);
            }
        }
    }
}

void buildBCT() {
    dfs_bct();
}
///////////////////////////////////////////////////////////////////////////
@Den
// Small to large (number of different colors in subtree)

vll graph[N];

set <ll> *pointers[N];

ll col[N], res[N];

void dfs(ll a) {
    if (graph[a].empty()) {
        pointers[a] = new set <ll>();
        pointers[a]->insert(col[a]);

        res[a] = 1;

        return;
    }

    for (auto c : graph[a])
        dfs(c);

    ll cur = graph[a].front();

    for (auto c : graph[a])
        if (pointers[c]->size() > pointers[cur]->size())
            cur = c;

    pointers[a] = pointers[cur];

    for (auto c : graph[a])
        if (c != cur) {
            for (auto color : *pointers[c])
                pointers[a]->insert(color);
            delete pointers[c];
        }

    pointers[a]->insert(col[a]);

    res[a] = pointers[a]->size();
}
///////////////////////////////////////////////////////////////////////////
// @Den
// zip of strongly connected components
// code: https://codeforces.com/contest/1213/submission/92370737

const ll N = 2e5 + 5;

ll n;
ll comps = 0;

vll g[N], gr[N], gz[N]; // graph, reversed graph and zipped graph

ll used[N];
ll gl = 1;

vll order, component;

ll zipped[N];
vll unzipped[N];

void dfsTopSort(ll a) {
    used[a] = gl;
    for (auto c : g[a]) {
        if (used[c] != gl) {
            dfsTopSort(c);
        }
    }
    order.pb(a);
}

void dfsComponent(ll a) {
    used[a] = gl;
    
    component.pb(a);
    for (auto c : gr[a]) {
        if (used[c] != gl) {
            dfsComponent(c);
        }
    }
}

void addComponent(vll &component) {
    for (auto v : component) {
        zipped[v] = n + comps;
    }
    
    unzipped[n + comps] = component;
    
    comps++;
}

void zipGraph() {
    rep(a, 0, n) {
        if (used[a] != gl)
            dfsTopSort(a);
    }
    
    reverse(all(order));
    
    gl++;
    
    for (auto c : order) {
        if (used[c] != gl) {
            dfsComponent(c);
            
            addComponent(component);
            component.clear();
        }
    }
    
    rep(v, 0, n) {
        for (auto u : g[v]) {
            ll zu = zipped[u];
            ll zv = zipped[v];
            
            if (zv != zu) {
                gz[zv].pb(zu);
            }
        }
    }
}

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

////////////////////////////////////////////////////////////////////////////////////////////
s.sh by Karavaiev:

clang++ -std=c++11 -stdlib=libc++ ../MainSolution/main.cpp -o main
clang++ -std=c++11 -stdlib=libc++ ../BruteSolution/main.cpp -o brute
clang++ -std=c++11 -stdlib=libc++ ../Generator/main.cpp -o gen

for ((i = 1; ; ++i)); do
    echo $i
    {
        ./gen $i > int
        ./main < int > out1
    } || {
        break
    }
    ./brute < int > out2
    if !(cmp -s out1 out2); then
        echo -e "\nGAME OVER: $i\n"
        echo -e "test:\n" && cat int && echo -e ""
        echo -e "main:\n" && cat out1 && echo -e ""
        echo -e "brute:\n" && cat out2 && echo -e ""
        break
    fi
done

s.sh + Checker by Karavaiev:

clang++ -std=c++11 -stdlib=libc++ Doree/main.cpp -o main
clang++ -std=c++11 -stdlib=libc++ Generator/main.cpp -o gen
clang++ -std=c++11 -stdlib=libc++ BruteSolution/main.cpp -o brute
clang++ -std=c++11 -stdlib=libc++ Checker/main.cpp -o check
 
for ((i = 1; ; ++i)); do
    echo $i
    {
        ./gen $i > int
        cat int > out
        ./main < int >> out
        ./brute < int >> out
    } || {
        break
    }
    ./check < out
    if [ $? -eq 0 ]
    then
      echo "Successfully created file"
      cat out
    else
      cat out
      break
    fi
done
