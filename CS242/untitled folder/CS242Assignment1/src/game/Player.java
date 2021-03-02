package game;

import unocard.PlayerCard;
import unocard.UnoCard;

public class Player {
	
	private int pid;
	private String pname;
	private PlayerCard pcard;
	
	/**
	 * pid: The ID number of the player starting from 0
	 * pname: The name of the player
	 * pcard: The Uno Cards in the player's hand
	 */
	public Player(int pid, String name) {
		this.pid = pid;
		this.pname = name;
		this.pcard = new PlayerCard();
	}
	

	/**
	 * @return the name of the player
	 */
	public String getPname() {
		return this.pname;
	}
	
	
	/**
	 * @return the ID number of the player
	 */
	public int getPid() {
		return this.pid;
	}
	
	
	/**
	 * add a Uno Card to the player's hand
	 * @param newCard: the Uno Card to be added
	 */
	public void addCard(UnoCard newCard) {
		this.pcard.addCard(newCard);
	}
	
	/**
	 * Get the number of Uno Cards in the player's hand;
	 * @return number of Uno Cards in hand;
	 */
	public int getNumCards() {
		return this.pcard.getNumUnoCards();
	}
	
	
	/**
	 * play the Uno Card with the given index
	 * @param idx: the index of the card in the player's hand
	 * @return the Uno Card played
	 */
	public UnoCard playCard(int idx) {
		UnoCard card = this.pcard.playCard(idx);
		return card;
	}
	
	
	/**
	 * play the Uno Card with the given name
	 * @param idx: the name of the card in the player's hand. (Color+ [Num / Action])
	 * @return the Uno Card played
	 */
	public UnoCard playCard(String name) {
		
		for (int i = 0; i < this.pcard.getNumUnoCards(); i++) {
			if(this.pcard.getCard(i).equals(name)) {
				return this.pcard.playCard(i);
			}
		}
		return null;
	}
	
	
}
