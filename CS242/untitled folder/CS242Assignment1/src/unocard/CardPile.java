package unocard;

import java.util.ArrayList;

public class CardPile {
	
	protected ArrayList <UnoCard> cardPile;
	
	public CardPile() {
		this.cardPile = new ArrayList<UnoCard>();
	}

	
	/**
	 * Get the number of Uno Cards in the draw pile.
	 * @return number of Uno Cards in the draw pile.
	 */
	public int getNumUnoCards() {
		return this.cardPile.size();
	}
	
	
	/**
	 * Get the Uno Card with the given index.
	 * @return the Uno Card with the given index. Returns null if index out of bound.
	 */
	public UnoCard getCard(int idx) {
		try {
			return this.cardPile.get(idx);
		} catch(IndexOutOfBoundsException e) {
			return null;
		}
	}
	
	/**
	 * Get and Remove the Uno Card with the given index.
	 * @return the Uno Card with the given index. Returns null if index out of bound.
	 */
	public UnoCard removeCard(int idx) {
		try {
			return this.cardPile.remove(idx);
		} catch(IndexOutOfBoundsException e) {
			return null;
		}
	}
	
	/**
	 * Get the last card discarded (for discardPile).
	 * @return the top Uno Card.
	 */
	public UnoCard getNextCard() {
		int numOfCards = getNumUnoCards();
		UnoCard topCard = this.cardPile.get(numOfCards-1);
		return topCard;
	}
	
	/**
	 * Add a Uno Card to the top of draw pile.
	 * @param the Uno Card to be added.
	 */
	public void addCard(UnoCard card) {
		this.cardPile.add(card);
	}
	
	

}
