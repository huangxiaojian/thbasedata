#include "Vector.h"
#include "Matrix.h"
#include <vector>

//testing
#include "Transform.h"


#include "AutoProfile.h"

using namespace std;

void Print(__m256d a)
{
	cout << a.m256d_f64[0] << ' ' << a.m256d_f64[1] << ' ' << a.m256d_f64[2] << ' ' << a.m256d_f64[3] << endl << endl;
}

void Print(__m128 a)
{
	cout << a.m128_f32[0] << ' ' << a.m128_f32[1] << ' ' << a.m128_f32[2] << ' ' << a.m128_f32[3] << endl;
}

int main()
{
	////AUTO_PROFILE_BEGIN("SSE");
	//vector<Vector4> v;
	//Vector4 a(1, 2, 3);
	//Vector4 b(4, 5, 6);
	//Vector4 c = a + b;
	//v.push_back(a);
	//v.push_back(b);
	//c.print();
	//Vector4 d = v[0] + v[1];
	//d.print();
	//print(c.m_vec);
	//cout << a.dot(b) << endl;
	//cout << SIMD_SHUFFLE_MASK(1, 1, 1, 1) << endl;
	//print(a.m_vec);
	////print(SIMD_SHUFFLE(a.m_vec, SIMD_SHUFFLE_MASK(1, 1, 1, 1)));
	//print(SIMD_SHUFFLE(a.m_vec, SIMD_SHUFFLE_MASK(0, 0, 0, 0)));
	//print(SIMD_SHUFFLE(a.m_vec, SIMD_SHUFFLE_MASK(0, 1, 0, 0)));
	//print(SIMD_SHUFFLE(a.m_vec, SIMD_SHUFFLE_MASK(1, 0, 0, 0)));
	//print(_mm_shuffle_ps(a.m_vec, a.m_vec, 0x80));
	//print(_mm_shuffle_ps(a.m_vec, a.m_vec, SIMD_SHUFFLE_MASK(0, 0, 0, 2)));
	////AUTO_PROFILE_END;

	//__m128 vd = _mm_mul_ps(a.m_vec, b.m_vec);
	//__m128 z = _mm_movehl_ps(vd, vd);
	//__m128 y = _mm_shuffle_ps(vd, vd, 0x55);
	//vd = _mm_add_ss(vd, y);
	//vd = _mm_add_ss(vd, z);
	//Scalar f = _mm_cvtss_f32(vd);
	//
	
	/*Vector4 b = -a;
	b.print();*/

	Vector4 vector(2, 2, 2);
	Quaternion a(1,1,1,1);
	Quaternion b=a*2;
	Quaternion c(a.get());
	cout << "a\t"; a.print();
	cout << "b\t"; b.print();
	//cout << "c\t"; c.print();
	double scalar = 2;
	cout << "scalar\t"<<scalar << endl;
	//Quaternion d;
	
	
	c = a + b;
	cout << '+' << endl;
	c.print();

	c = a + b;
	cout << '-' << endl;
	c.print();

	c = a * b;
	cout << '*' << endl;
	c.print();

	c = a*scalar;
	cout <<"quat"<<'*'<<scalar << endl;
	c.print();

	c = a*vector;
	cout << "quater*vector" << endl;
	c.print();

	c = vector*a;
	cout << "vector*quater" << endl;
	c.print();


	scalar = a.angle(b);
	cout << "angle between a and b:\t" << scalar << endl;

	scalar = a.getAngle();
	cout << "angle of a:\t" << scalar << endl;

	scalar = a.dot(b);
	cout << "DOT a and b: \t" << scalar << endl;


	Vector4 axis = a.getAxis();
	cout << "axis of a:\t"; axis.print();

	const Quaternion d(-1,-1,-1,-1);
	c = *(const_cast<Quaternion*> (&d));
	c.print();

	c= a.inverse();
	cout << "inverse of a:\t"; c.print();

	scalar = a.length();
	cout << "a.length:\t" << scalar << endl;

	scalar = a.length2();
	cout << "a.length2:\t" << scalar << endl;

	/*c = a.normalize();
	cout << "a.normalize:\t" << endl; 
	a.print();
	c.print();*/

	/*c = a.normalized();
	cout << "a.normalized:\t" << endl;
	a.print();
	c.print();*/

	a.set(b.get());
	cout << "a.set(b):\t"; a.print();

	a.setEuler(0.3,0.3,0.3);
	cout << "a.setEuler:\t"; a.print();

	a.setEulerZYX(0.3, 0.3, 0.3);
	cout << "a.setEulerZYX():\t"; a.print();

	a.setRotation(Vector4(1, 1, 1), 0.3);
	cout << "a.Rotation():\t"; a.print();

	a.setValue(0.2, 0.2, 0.2, 0.2);
	cout << "a.setValue():\t"; a.print();

	a = a.slerp(Quaternion(0,0,0,1), 0.2);
	cout << "a.slerp(,0.2):\t"; a.print();




	if (a == b)
		cout << "a==b" << endl;

	a = a.getIdentity();
	a *= b;
	cout << "*=\t"; a.print();

	a = a.getIdentity();
	a /= scalar;

	a = a.getIdentity();
	a += b;

	a = a.getIdentity();
	a -= b;



	Vector4 my(0, 0, 0);
	Vector4 myo(2, 3, 4, 0);
	my.normalize().print();


	getchar();
	return 0;
}