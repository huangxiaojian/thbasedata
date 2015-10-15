/////////////////////////////////////////////////////////////////////////////////
///  Copyright (C), 2015-?, Xiaojian Huang
///  \file  Vector.h
///  \brief Vector module in SIMD library
///  \details Vector related functions and SSE and AVX instruction micro
///  \author Xiaojian Huang
///   \version 1.0
///   \date    2015.5
///   \History: 
///                
///      Xiaojian Huang   2015.5     1.0     Create this file 
///  \warning	All rights reserved.
///  Redistribution and use in source and binary forms, with or without
///  modification, are permitted provided that the following conditions
///  are met:
///     1. Redistributions of source code must retain the above copyright
///        notice, this list of conditions and the following disclaimer.
///     2. Redistributions in binary form must reproduce the above copyright
///        notice, this list of conditions and the following disclaimer in the
///        documentation and/or other materials provided with the distribution.
///     3. The names of its contributors may not be used to endorse or promote
///        products derived from this software without specific prior written
///        permission.
///  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
///  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
///  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
///  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
///  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
///  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
///  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
///  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
///  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
///	 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
///	 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////////
#ifndef TH_VECTORS_H
#define TH_VECTORS_H

#include "Scalar.h"

ATTRIBUTE_ALIGNED(class) Vector4
{
public:
	union {
		Scalar4	m_vec;
		Scalar m_scalar[4];
	};

	TH_FORCE_INLINE Scalar4 get() const { return m_vec; }
	TH_FORCE_INLINE void set(Scalar4& v){ m_vec = v; }

	TH_FORCE_INLINE explicit Vector4()
	{
#ifdef USE_SSE
		m_vec = SIMD_TO_VEC(1.0, 1.0, 1.0);
#else
		m_scalar[0] = 1.0;
		m_scalar[1] = 1.0;
		m_scalar[2] = 1.0;
		m_scalar[3] = 0.0;
#endif
	}

	TH_FORCE_INLINE explicit Vector4(const Scalar x, const Scalar y)
	{
#ifdef USE_SSE
		m_vec = SIMD_TO_VEC(x, y, 0.0);
#else
		m_scalar[0] = x;
		m_scalar[1] = y;
		m_scalar[2] = 0.0;
		m_scalar[3] = 0.0;
#endif
	}

	TH_FORCE_INLINE explicit Vector4(const Scalar x, const Scalar y, const Scalar z)
	{
#ifdef USE_SSE
		m_vec = SIMD_TO_VEC(x, y, z);
#else
		m_scalar[0] = x;
		m_scalar[1] = y;
		m_scalar[2] = z;
		m_scalar[3] = 0.0;
#endif
	}

	TH_FORCE_INLINE explicit Vector4(const Scalar x, const Scalar y, const Scalar z, const Scalar w)
	{
#ifdef USE_SSE
		m_vec = SIMD_SET(x, y, z, w);
#else
		m_scalar[0] = x;
		m_scalar[1] = y;
		m_scalar[2] = z;
		m_scalar[3] = w;
#endif
	}

	TH_FORCE_INLINE Vector4(const Vector4& other){ m_vec = other.m_vec; }
	TH_FORCE_INLINE explicit Vector4(const Scalar4& other){ m_vec = other; }
	TH_FORCE_INLINE explicit Vector4(const Scalar* m)
	{
#ifdef USE_SSE
		m_vec = SIMD_SET(m[0], m[1], m[2], 0.0);
#else
		m_scalar[0] = m[0];
		m_scalar[1] = m[1];
		m_scalar[2] = m[2];
		m_scalar[3] = Scalar(0.0);
#endif
	}

	friend Vector4 operator+(const Vector4& v1, const Vector4& v2);
	friend Vector4 operator-(const Vector4& v1, const Vector4& v2);
	friend Vector4 operator*(const Vector4& v1, const Vector4& v2);
	friend Vector4 operator/(const Vector4& v1, const Vector4& v2);
	friend Vector4 operator*(const Vector4& v1, const Scalar v2);
	friend Vector4 operator/(const Vector4& v1, const Scalar v2);
	friend Vector4 operator-(const Vector4& v);

	TH_FORCE_INLINE Vector4& operator=(const Vector4& other)
	{
		m_vec = other.m_vec;
		return *this;
	}

	TH_FORCE_INLINE Vector4& operator=(const Scalar4& other)
	{
		m_vec = other;
		return *this;
	}

	/**@brief Return the ith member of this vector
	*/
	TH_FORCE_INLINE Scalar& operator[](int i)
	{
		return m_scalar[i];
	}

	/**@brief Return the ith member of this vector
	*/
	TH_FORCE_INLINE const Scalar& operator[](int i) const
	{
		return m_scalar[i];
	}

	/**@brief Return the negative of this vector
	* This simply negates each element */
	/*TH_FORCE_INLINE Vector4& operator-()
	{
#ifdef USE_SSE
		m_vec = SIMD_XOR(m_vec, SIMDZEROMASK);
		return *this;
#else
		m_scalar[0] = -m_scalar[0];
		m_scalar[1] = -m_scalar[1];
		m_scalar[2] = -m_scalar[2];
		return *this;
#endif
	}*/

	TH_FORCE_INLINE Vector4 operator-=(const Vector4& other)
	{
#ifdef USE_SSE
		m_vec = SIMD_SUB(m_vec, other.m_vec);
#else
		m_scalar[0] -= other.m_scalar[0];
		m_scalar[1] -= other.m_scalar[1];
		m_scalar[2] -= other.m_scalar[2];
#endif
		return *this;
	}

	TH_FORCE_INLINE Vector4& operator+=(const Vector4& other)
	{
#ifdef USE_SSE
		m_vec = SIMD_ADD(m_vec, other.m_vec);
#else
		m_scalar[0] += other.m_scalar[0];
		m_scalar[1] += other.m_scalar[1];
		m_scalar[2] += other.m_scalar[2];
#endif
		return *this;
	}

	TH_FORCE_INLINE Vector4 operator*=(const Vector4& other)
	{
#ifdef USE_SSE
		m_vec = SIMD_MUL(m_vec, other.m_vec);
#else
		m_scalar[0] *= other.m_scalar[0];
		m_scalar[1] *= other.m_scalar[1];
		m_scalar[2] *= other.m_scalar[2];
#endif
		return *this;
	}

	TH_FORCE_INLINE Vector4 operator/=(const Vector4& other)
	{
#ifdef USE_SSE
		Scalar4 v4 = SIMD_SET(other.m_scalar[0], other.m_scalar[1], other.m_scalar[2], 1);
		m_vec = SIMD_DIV(m_vec, other.m_vec);
#else
		m_scalar[0] /= other.m_scalar[0];
		m_scalar[1] /= other.m_scalar[1];
		m_scalar[2] /= other.m_scalar[2];
#endif
		return *this;
	}

	TH_FORCE_INLINE Vector4 operator*=(const Scalar& v)
	{
#ifdef USE_SSE
		Vector4 other(v, v, v);
		m_vec = SIMD_MUL(m_vec, other.m_vec);
#else
		m_scalar[0] *= v;
		m_scalar[1] *= v;
		m_scalar[2] *= v;
#endif
		return *this;
	}

	TH_FORCE_INLINE Vector4 operator/=(const Scalar& v)
	{
#ifdef USE_SSE
		Scalar4 v4 = SIMD_SET(v, v, v, 1);
		m_vec = SIMD_DIV(m_vec, v4);
#else
		m_scalar[0] /= v;
		m_scalar[1] /= v;
		m_scalar[2] /= v;
#endif
		return *this;
	}

	/**@brief Return the dot product
	 * @param other: The other vector in the dot product */
	TH_FORCE_INLINE Scalar dot(const Vector4& other) const
	{
#ifdef USE_SSE
		Scalar4 vd = SIMD_MUL(m_vec, other.m_vec);
		return SIMD_V_SUM(vd);
#else
		return m_scalar[0] * other.m_scalar[0] + m_scalar[1] * other.m_scalar[1] + m_scalar[2] * other.m_scalar[2];
#endif
	}

	/**@brief Return the length of the vector squared */
	TH_FORCE_INLINE Scalar length2() const
	{
		return dot(*this);
	}

	/**@brief Return the length of the vector */
	TH_FORCE_INLINE Scalar length() const
	{
		return thSqrt(length2());
	}

	/**@brief Return the norm (length) of the vector */
	TH_FORCE_INLINE Scalar norm() const
	{
		return length();
	}

	//TODO
	/**@brief Return a vector will the absolute values of each element */
	Vector4 absolute() const
	{
		return Vector4(thFabs(m_scalar[0]), thFabs(m_scalar[1]), thFabs(m_scalar[2]));
	}

	/**@brief Return the distance squared between the ends of this and another vector
	* This is symantically treating the vector like a point */
	Scalar distance2(const Vector4& other) const
	{
		return (*this - other).length2();
	}

	/**@brief Return the distance between the ends of this and another vector
	* This is symantically treating the vector like a point */
	Scalar distance(const Vector4& other) const
	{
		return (*this - other).length();
	}

	//TODO
	/**@brief Normalize this vector
	* x^2 + y^2 + z^2 = 1 */
	Vector4& normalize()
	{
		Scalar l = length();
		l = l <= TH_EPSILON ? 1 : l;
		return (*this) /= l;
	}

	/**@brief Return a normalized version of this vector */
	Vector4 normalized() const
	{
		Vector4 tmp = *this;
		return tmp.normalize();
	}

	//TODO
	//rotate direction: curVector.cross(wAxis)
	/**@brief Return a rotated version of this vector
	* @param wAxis The axis to rotate about
	* @param angle The angle to rotate by */
	Vector4 rotate(const Vector4& wAxis, const Scalar angle) const
	{
#ifdef USE_SSE
		Scalar4 O = SIMD_MUL(wAxis.m_vec, m_vec);
		Scalar ssin = thSin( angle );
		Scalar4 C = wAxis.cross( *this ).m_vec;
		O = SIMD_AND(O, SIMDFFF0MASK);
		Scalar scos = thCos( angle );

		Scalar4 vsin = SIMD_SET(ssin, 0, 0, 0);	//	(S 0 0 0)
		Scalar4 vcos = SIMD_SET(scos, 0, 0, 0);	//	(S 0 0 0)

		Scalar4 Y = SIMD_SHUFFLE(O, 0xC9);	//	(Y Z X 0)
		Scalar4 Z = SIMD_SHUFFLE(O, 0xD2);	//	(Z X Y 0)
		O = SIMD_ADD(O, Y);
		vsin = SIMD_SHUFFLE(vsin, 0x80);	//	(S S S 0)
		O = SIMD_ADD(O, Z);
		vcos = SIMD_SHUFFLE(vcos, 0x80);	//	(S S S 0)

		vsin = SIMD_MUL(vsin, C);
		O = SIMD_MUL(O, wAxis.m_vec);
		Scalar4 X = SIMD_SUB(m_vec, O);

		O = SIMD_ADD(O, vsin);
		vcos = SIMD_MUL(vcos, X);
		O = SIMD_ADD(O, vcos);

		return Vector4(O);
#else
		Vector4 o = wAxis * wAxis.dot(*this);
		Vector4 _x = *this - o;
		Vector4 _y;

		_y = wAxis.cross(*this);
		
		return (o + _x * thCos(angle) + _y * thSin(angle));
#endif
	}
	
	/**@brief Return the angle between this and another vector
	* @param other The other vector */
	Scalar angle(const Vector4& other) const
	{
		Scalar l1 = length2();
		if (l1 <= TH_EPSILON)
			return TH_PI;
		Scalar l2 = other.length2();
		if (l2 <= TH_EPSILON)
			return TH_PI;
		Scalar s = thSqrt(l1 * l2);
		return thAcos(dot(other) / s);
	}

	/**@brief Return the cross product between this and another vector
	* @param other The other vector */
	TH_FORCE_INLINE Vector4 cross(const Vector4& other) const
	{
#ifdef USE_SSE
		Scalar4	T, V;
		T = SIMD_SHUFFLE(m_vec, SIMD_SHUFFLE_MASK(1, 2, 0, 3));
		V = SIMD_SHUFFLE(other.m_vec, SIMD_SHUFFLE_MASK(1, 2, 0, 3));

		V = SIMD_MUL(V, m_vec);
		T = SIMD_MUL(T, other.m_vec);
		V = SIMD_SUB(V, T);

		V = SIMD_SHUFFLE(V, SIMD_SHUFFLE_MASK(1, 2, 0, 3));

		return Vector4(V);
#else
		return Vector4(
			m_scalar[1] * other.m_scalar[2] - m_scalar[2] * other.m_scalar[1],
			m_scalar[2] * other.m_scalar[0] - m_scalar[0] * other.m_scalar[2],
			m_scalar[0] * other.m_scalar[1] - m_scalar[1] * other.m_scalar[0]
			);
#endif
	}

	//TODO
	Scalar triple(const Vector4& v1, const Vector4& v2) const
	{
		//return dot(v1.cross(v2));
#ifdef USE_SSE
		Scalar4	T, V;
		T = SIMD_SHUFFLE(v1.m_vec, SIMD_SHUFFLE_MASK(1, 2, 0, 3));
		V = SIMD_SHUFFLE(v2.m_vec, SIMD_SHUFFLE_MASK(1, 2, 0, 3));

		V = SIMD_MUL(V, v1.m_vec);
		T = SIMD_MUL(T, v2.m_vec);
		V = SIMD_SUB(V, T);

		V = SIMD_SHUFFLE(V, SIMD_SHUFFLE_MASK(1, 2, 0, 3));
		T = SIMD_MUL(m_vec, V);
		return SIMD_V_SUM(T);
#else
		return
			m_scalar[0] * (v1.m_scalar[1] * v2.m_scalar[2] - v1.m_scalar[2] * v2.m_scalar[1]) +
			m_scalar[1] * (v1.m_scalar[2] * v2.m_scalar[0] - v1.m_scalar[0] * v2.m_scalar[2]) +
			m_scalar[2] * (v1.m_scalar[0] * v2.m_scalar[1] - v1.m_scalar[1] * v2.m_scalar[0]);
#endif
	}

	/**@brief Return the axis with the smallest value
	* Note return values are 0,1,2 for x, y, or z */
	TH_FORCE_INLINE int minAxis() const
	{
		return m_scalar[0] < m_scalar[1] ? (m_scalar[0] < m_scalar[2] ? 0 : 2) : (m_scalar[1] < m_scalar[2] ? 1 : 2);;
	}

	/**@brief Return the axis with the largest value
	* Note return values are 0,1,2 for x, y, or z */
	TH_FORCE_INLINE int maxAxis() const
	{
		return m_scalar[0] < m_scalar[1] ? (m_scalar[1] <m_scalar[2] ? 2 : 1) : (m_scalar[0] <m_scalar[2] ? 2 : 0);
	}

	TH_FORCE_INLINE int furthestAxis() const
	{
		return absolute().minAxis();
	}

	TH_FORCE_INLINE int closestAxis() const
	{
		return absolute().maxAxis();
	}

	//TODO
	void setInterpolate(const Vector4& v0, const Vector4& v1, Scalar t)
	{
		Scalar s = Scalar(1.0) - t;
#ifdef USE_SSE
		Scalar4 sv = SIMD_TO_VEC(s, s, s);
		Scalar4 tv = SIMD_TO_VEC(t, t, t);
		m_vec = SIMD_ADD(SIMD_MUL(v0.m_vec,sv), SIMD_MUL(v1.m_vec,tv));
#else
		m_scalar[0] = s * v0.m_scalar[0] + t * v1.m_scalar[0];
		m_scalar[1] = s * v0.m_scalar[1] + t * v1.m_scalar[1];
		m_scalar[2] = s * v0.m_scalar[2] + t * v1.m_scalar[2];
#endif
	}

	/** @brief Set from the translation part of a 4x4 OpenGL matrix
	*  @param m A pointer to the beginning of the array of scalars*/
	void setFromOpenGLMatrixTranslationPart(const Scalar* m)
	{
		setValue(m[12], m[13], m[14]);
	}

	/** @brief Fill the translation part of a 4x4 OpenGL matrix
	*  @param m A pointer to the beginning of the array of scalars*/
	void getOpenGLMatrixTranslationPart(Scalar* m)
	{
		m[12] = m_scalar[0];
		m[13] = m_scalar[1];
		m[14] = m_scalar[2];
	}

	//TODO
	/**@brief Return the linear interpolation between this and another vector
	* @param v The other vector
	* @param t The ration of this to v (t = 0 => return this, t=1 => return other) */
	Vector4 lerp(const Vector4& v, const Scalar t) const
	{
#ifdef USE_SSE
		Scalar4 tv = SIMD_TO_VEC(t, t, t);
		return Vector4(SIMD_ADD(m_vec, SIMD_MUL(SIMD_SUB(v.m_vec, m_vec), tv)));
#else
		return
			Vector4(m_scalar[0] + (v.m_scalar[0] - m_scalar[0]) * t,
			m_scalar[1] + (v.m_scalar[1] - m_scalar[1]) * t,
			m_scalar[2] + (v.m_scalar[2] - m_scalar[2]) * t);
#endif
	}

	/**@brief Return the x value */
	TH_FORCE_INLINE const Scalar& x() const { return m_scalar[0]; }
	/**@brief Return the y value */
	TH_FORCE_INLINE const Scalar& y() const { return m_scalar[1]; }
	/**@brief Return the z value */
	TH_FORCE_INLINE const Scalar& z() const { return m_scalar[2]; }
	/**@brief Return the w value */
	//TH_FORCE_INLINE const Scalar& w() const { return m_scalar[3]; }


	//TODO
	bool operator==(const Vector4& other) const
	{
		return ((m_scalar[3] == other.m_scalar[3]) &&
			(m_scalar[2] == other.m_scalar[2]) &&
			(m_scalar[1] == other.m_scalar[1]) &&
			(m_scalar[0] == other.m_scalar[0]));
	}

	//TODO
	bool operator!=(const Vector4& other) const
	{
		return !(*this == other);
	}
	
	//TODO
	/**@brief Set each element to the max of the current values and the values of another Vector4
	* @param other The other Vector4 to compare with
	*/
	void setMax(const Vector4& other)
	{
#ifdef USE_SSE
		m_vec = SIMD_MAX(m_vec, other.m_vec);
#else
		thSetMax(m_scalar[0], other.m_scalar[0]);
		thSetMax(m_scalar[1], other.m_scalar[1]);
		thSetMax(m_scalar[2], other.m_scalar[2]);
		thSetMax(m_scalar[3], other.m_scalar[3]);
#endif
	}

	//TODO
	/**@brief Set each element to the min of the current values and the values of another Vector4
	* @param other The other Vector4 to compare with
	*/
	void setMin(const Vector4& other)
	{
#ifdef USE_SSE
		m_vec = SIMD_MIN(m_vec, other.m_vec);
#else
		thSetMin(m_scalar[0], other.m_scalar[0]);
		thSetMin(m_scalar[1], other.m_scalar[1]);
		thSetMin(m_scalar[2], other.m_scalar[2]);
		thSetMin(m_scalar[3], other.m_scalar[3]);
#endif
	}

	/**@brief Set the values
	* @param x Value of x
	* @param y Value of y
	* @param z Value of z
	* @param w Value of w
	*/
	void setValue(const Scalar& _x, const Scalar& _y, const Scalar& _z)
	{
		m_scalar[0] = _x;
		m_scalar[1] = _y;
		m_scalar[2] = _z;
		m_scalar[3] = Scalar(0);
	}

	//void getSkewSymmetricMatrix(Vector4* v0, Vector4* v1, Vector4* v2) const;

	void setZero()
	{
		m_vec = SIMD_XOR(m_vec, m_vec);
	}

	bool fuzzyZero() const
	{
		return length2() < TH_EPSILON * TH_EPSILON;
	}

	bool isZero() const
	{
		return m_scalar[0] == Scalar(0) && m_scalar[1] == Scalar(0) && m_scalar[2] == Scalar(0);
	}

	TH_FORCE_INLINE void print()
	{
		std::cout << m_scalar[0] << ", " << m_scalar[1]
			<< ", " << m_scalar[2] << std::endl;
	}

	std::string toString()
	{
		std::ostringstream os;
		os << "(" << m_scalar[0] << ", " << m_scalar[1]
			<< ", " << m_scalar[2] << ")" << std::endl;
		return os.str();
	}

	/* create a vector as  btVector3( this->dot( btVector3 v0 ), this->dot( btVector3 v1), this->dot( btVector3 v2 ))  */
	TH_FORCE_INLINE Vector4  dot3(const Vector4 &v0, const Vector4 &v1, const Vector4 &v2) const
	{
		return Vector4(dot(v0), dot(v1), dot(v2));
	}


};

typedef Vector4 thPoint;
typedef Vector4 Vector3;

//friend function for Vector4
//+, -, *, /, *(scalar), /(scalar)
/**@brief Return the negative of the vector */
Vector4 operator-(const Vector4& v)
{
#ifdef USE_SSE
	Scalar4 r = SIMD_XOR(v.m_vec, SIMDZEROMASK);
	return Vector4(SIMD_AND(r, SIMDFFF0MASK));
#else	
	return Vector4(-v.m_scalar[0], -v.m_scalar[1], -v.m_scalar[2]);
#endif
}

TH_FORCE_INLINE Vector4 operator+(const Vector4& v1, const Vector4& v2)
{
#ifdef USE_SSE
	Scalar4 vec = SIMD_ADD(v1.m_vec, v2.m_vec);
	return Vector4(vec);
#else
	return Vector4(v1.m_scalar[0] + v2.m_scalar[0], v1.m_scalar[1] + v2.m_scalar[1],
		v1.m_scalar[2] + v2.m_scalar[2]);
#endif
}

TH_FORCE_INLINE Vector4 operator-(const Vector4& v1, const Vector4& v2)
{
#ifdef USE_SSE
	Scalar4 vec = SIMD_SUB(v1.m_vec, v2.m_vec);
	return Vector4(vec);
#else
	return Vector4(v1.m_scalar[0] - v2.m_scalar[0], v1.m_scalar[1] - v2.m_scalar[1],
		v1.m_scalar[2] - v2.m_scalar[2]);
#endif
}

TH_FORCE_INLINE Vector4 operator*(const Vector4& v1, const Vector4& v2)
{
#ifdef USE_SSE
	Scalar4 vec = SIMD_MUL(v1.m_vec, v2.m_vec);
	return Vector4(vec);
#else
	return Vector4(v1.m_scalar[0] * v2.m_scalar[0], v1.m_scalar[1] * v2.m_scalar[1],
		v1.m_scalar[2] * v2.m_scalar[2]);
#endif
}

TH_FORCE_INLINE Vector4 operator/(const Vector4& v1, const Vector4& v2)
{
#ifdef USE_SSE
	Scalar4 vec = SIMD_DIV(v1.m_vec, v2.m_vec);
	return Vector4(vec);
#else
	return Vector4(v1.m_scalar[0] / v2.m_scalar[0], v1.m_scalar[1] / v2.m_scalar[1],
		v1.m_scalar[2] / v2.m_scalar[2]);
#endif
}

TH_FORCE_INLINE Vector4 operator*(const Vector4& v1, const Scalar v2)
{
#ifdef USE_SSE
	Vector4 v(v2, v2, v2);
	Scalar4 vec = SIMD_MUL(v1.m_vec, v.m_vec);
	return Vector4(vec);
#else
	return Vector4(v1.m_scalar[0] * v2, v1.m_scalar[1] * v2, v1.m_scalar[2] * v2);
#endif
}

TH_FORCE_INLINE Vector4 operator/(const Vector4& v1, const Scalar v2)
{
#ifdef USE_SSE
	Vector4 v(v2, v2, v2);
	Scalar4 vec = SIMD_DIV(v1.m_vec, v.m_vec);
	return Vector4(vec);
#else
	return Vector4(v1.m_scalar[0] / v2, v1.m_scalar[1] / v2, v1.m_scalar[2] / v2);
#endif
}

/**@brief Return the dot product between two vectors */
Scalar thDot(const Vector4& v1, const Vector4& v2)
{
	return v1.dot(v2);
}


/**@brief Return the distance squared between two vectors */
Scalar thDistance2(const Vector4& v1, const Vector4& v2)
{
	return v1.distance2(v2);
}


/**@brief Return the distance between two vectors */
Scalar thDistance(const Vector4& v1, const Vector4& v2)
{
	return v1.distance(v2);
}

/**@brief Return the angle between two vectors */
Scalar thAngle(const Vector4& v1, const Vector4& v2)
{
	return v1.angle(v2);
}

/**@brief Return the cross product of two vectors */
Vector4 thCross(const Vector4& v1, const Vector4& v2)
{
	return v1.cross(v2);
}

Scalar thTriple(const Vector4& v1, const Vector4& v2, const Vector4& v3)
{
	return v1.triple(v2, v3);
}

/**@brief Return the linear interpolation between two vectors
* @param v1 One vector
* @param v2 The other vector
* @param t The ration of this to v (t = 0 => return v1, t=1 => return v2) */
Vector4 lerp(const Vector4& v1, const Vector4& v2, const Scalar& t)
{
	return v1.lerp(v2, t);
}

#endif //TH_VECTORS_H